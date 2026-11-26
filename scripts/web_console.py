# -*- coding: utf-8 -*-
# 毛发异物检测 · 树莓派网页控制台
# 界面与功能对齐 Windows 版 hair_detect_app.py：
#   图片 / 视频 / 摄像头检测，置信度与推理尺寸运行中可调，暂停/停止，
#   日志、记录表、结果保存（results\）、输出目录浏览
# 用法: ~/yolo-env/bin/python web_console.py [--port 8000]
import argparse
import csv
import glob
import io
import os
import re
import subprocess
import threading
import time
import zipfile
from collections import deque

import cv2
import numpy as np
import ncnn
from flask import Flask, Response, request, jsonify, send_from_directory, send_file, render_template_string

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "hair_yolov8n_v3_20261016_fp16_ncnn_model")
RESULT_DIR = os.path.join(BASE, "results")
UPLOAD_DIR = os.path.join(BASE, "uploads")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 上传上限 1GB

CONFIG = {"imgsz": 1280, "port": 8000}
STATE = {
    "view": None, "view_id": 0,
    "count": 0, "ms": 0.0, "fps": 0.0, "temp": "--",
    "conf": 0.45, "imgsz": 1280,
    "mode": "idle",            # idle / camera / images / video
    "display_off": False,      # 关闭摄像头画面显示（检测继续）
    "paused": False,
    "camera_dev": 0,
    "net_ready": False,
    "reject_en": True,         # 剔除模拟启用开关
    "reject_cnt": 0,           # 剔除计数（检出"无→有"的上升沿累计）
    "reject_active": False,    # 当前检出状态（有毛发=True）
    "last_hair": False,        # 上一帧/上一张图的检出状态
    "log": [],                 # ["HH:MM:SS 文本", ...]
    "records": [],             # [文件, 类型, 毛发数量, 耗时ms]
    "lock": threading.RLock(),
    "ms_history": [],
}
ACTIONS = deque()              # ("camera", dev) / ("images", [路径]) / ("video", 路径) / ("stop",)
ACTION_LOCK = threading.Lock()


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + str(msg)
    with STATE["lock"]:
        STATE["log"].append(line)
        if len(STATE["log"]) > 300:
            del STATE["log"][:-300]
    print(line, flush=True)


def push_view(img):
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    with STATE["lock"]:
        if ok:
            STATE["view"] = jpg.tobytes()
            STATE["view_id"] += 1


def placeholder(text):
    # 注意：OpenCV 的 putText 不支持中文，占位图文字一律用英文，避免显示成 ????
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, text, (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, lineType=cv2.LINE_AA)
    push_view(img)


def pending_action():
    # 会话内检查：有新任务排队即返回 True（不打断弹出，由主循环接管）
    with ACTION_LOCK:
        return bool(ACTIONS)


def stop_requested():
    # 仅当队首是「停止」时返回 True；文件/视频任务排队不打断当前会话
    with ACTION_LOCK:
        return bool(ACTIONS) and ACTIONS[0][0] == "stop"


ACT_HELPER = "/usr/local/bin/act-led"
LED_HOLD = 1.0              # 检出结束后指示灯保持时长（秒），避免单张图检测一闪而过
_led_timer = None


def _led_off():
    global _led_timer
    with STATE["lock"]:
        STATE["reject_active"] = False
    act_led(False)
    _led_timer = None


def act_led(on):
    # 控制板载 ACT LED（sudoers 已对树莓派控制台放行免密调用）
    try:
        subprocess.run(["sudo", "-n", ACT_HELPER, "on" if on else "off"],
                       check=False, timeout=3,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def apply_reject(count, per_item=False):
    # 剔除模拟：检出毛发（数量>0）→ 网页圆灯变红 + 板载 ACT 灯常亮；无检出全灭。
    # 计数规则：图片模式 per_item=True 每张有毛发的图 +1；视频/摄像头按"无→有"上升沿累计（连续检出算一次）。
    # 检出结束后指示灯保持 LED_HOLD 秒再熄灭（保持期间新检出会取消熄灭）。
    global _led_timer
    with STATE["lock"]:
        en = STATE["reject_en"]
        prev = STATE["last_hair"]
        was_active = STATE["reject_active"]
        has = count > 0
        if not en:
            if _led_timer:
                _led_timer.cancel()
                _led_timer = None
            if prev or was_active:
                STATE["last_hair"] = False
                STATE["reject_active"] = False
                act_led(False)
            return
        if has:
            if per_item:
                STATE["reject_cnt"] += count   # 图片模式：按该图检出的毛发数量累加
                log("剔除动作触发（检出毛发 %d 根）" % count)
            elif not prev:
                STATE["reject_cnt"] += 1
                log("剔除动作触发（检出毛发）")
            STATE["last_hair"] = True
            STATE["reject_active"] = True
        else:
            STATE["last_hair"] = False
            if _led_timer:
                _led_timer.cancel()
            _led_timer = threading.Timer(LED_HOLD, _led_off)
            _led_timer.daemon = True
            _led_timer.start()
    if en and has and not was_active:
        act_led(True)


def pop_action():
    with ACTION_LOCK:
        return ACTIONS.popleft() if ACTIONS else None


# ---------- NCNN 推理核心（沿用 2026-10-14 已验证实现） ----------
def letterbox(img, newsize):
    h, w = img.shape[:2]
    r = min(newsize / h, newsize / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh))
    dw, dh = (newsize - nw) / 2, (newsize - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def nms(bxy, sc, thr=0.45):
    order = sc.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(bxy[i, 0], bxy[order[1:], 0])
        yy1 = np.maximum(bxy[i, 1], bxy[order[1:], 1])
        xx2 = np.minimum(bxy[i, 2], bxy[order[1:], 2])
        yy2 = np.minimum(bxy[i, 3], bxy[order[1:], 3])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        a1 = (bxy[i, 2] - bxy[i, 0]) * (bxy[i, 3] - bxy[i, 1])
        a2 = (bxy[order[1:], 2] - bxy[order[1:], 0]) * (bxy[order[1:], 3] - bxy[order[1:], 1])
        ov = inter / (a1 + a2 - inter + 1e-9)
        order = order[1:][ov < thr]
    return keep


NET = None


def load_net():
    global NET
    net = ncnn.Net()
    try:
        net.opt.num_threads = 4
    except Exception:
        pass
    net.load_param(os.path.join(MODEL_DIR, "model.ncnn.param"))
    net.load_model(os.path.join(MODEL_DIR, "model.ncnn.bin"))
    NET = net


def detect(img, conf, imgsz):
    lb, r, dx, dy = letterbox(img, imgsz)
    x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x)
    ex = NET.create_extractor()
    ex.input("in0", ncnn.Mat(x))
    res = ex.extract("out0")
    m = res[1] if isinstance(res, tuple) else res
    out = np.array(m)
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T
    sc = out[:, 4]
    sel = out[sc >= conf]
    dets = []
    if len(sel):
        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        bxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        sc2 = sel[:, 4]
        k = nms(bxy, sc2)
        bxy, sc2 = bxy[k], sc2[k]
        bxy[:, [0, 2]] = (bxy[:, [0, 2]] - dx) / r
        bxy[:, [1, 3]] = (bxy[:, [1, 3]] - dy) / r
        dets = [(tuple(map(float, b)), float(s)) for b, s in zip(bxy, sc2)]
    return dets


def draw(img, dets):
    vis = img.copy()
    for (x1, y1, x2, y2), s in dets:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(vis, "hair %.2f" % s, (int(x1), max(28, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
    count = len(dets)
    color = (0, 0, 255) if count > 0 else (0, 255, 0)
    cv2.putText(vis, "Hair Count: %d" % count, (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 4, lineType=cv2.LINE_AA)
    return vis


def infer_one(img):
    t0 = time.perf_counter()
    try:
        dets = detect(img, STATE["conf"], CONFIG["imgsz"])
    except Exception as exc:
        # 个别推理尺寸不被 NCNN 支持（导出固定 1280）时回退，不让线程崩
        log("[警告] 推理尺寸 %d 推理失败（%s），已回退 1280" % (CONFIG["imgsz"], exc))
        CONFIG["imgsz"] = 1280
        with STATE["lock"]:
            STATE["imgsz"] = 1280
        dets = detect(img, STATE["conf"], CONFIG["imgsz"])
    ms = (time.perf_counter() - t0) * 1000
    vis = draw(img, dets)
    return vis, dets, ms


def update_stats(count, ms):
    with STATE["lock"]:
        STATE["count"] = count
        STATE["ms"] = ms
        hist = STATE["ms_history"]
        hist.append(ms)
        if len(hist) > 12:
            hist.pop(0)
        avg = sum(hist) / len(hist)
        STATE["fps"] = 1000.0 / avg if avg > 0 else 0.0


def record_row(name, typ, count, ms, csv_path):
    # 每轮检测的报表保存在各自会话目录内，互不追加
    with STATE["lock"]:
        STATE["records"].append([name, typ, str(count), "%.0f" % ms])
        if len(STATE["records"]) > 300:
            del STATE["records"][:-300]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        if new_file:
            writer.writerow(["文件", "类型", "毛发数量", "推理耗时ms"])
        writer.writerow([name, typ, count, "%.0f" % ms])


# ---------- 三种检测会话 ----------
def run_camera(dev):
    with STATE["lock"]:
        STATE["mode"] = "camera"
        STATE["display_off"] = False
        STATE["records"] = []          # 新会话：清空检测记录表
        STATE["reject_cnt"] = 0        # 新会话：剔除计数归零
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        log("[错误] 无法打开摄像头 /dev/video%d" % dev)
        with STATE["lock"]:
            STATE["mode"] = "idle"
        placeholder("Cannot open camera /dev/video%d" % dev)
        return
    log("摄像头已打开（/dev/video%d），实时检测中..." % dev)
    placeholder_once = False
    while True:
        if pending_action():
            break
        with STATE["lock"]:
            paused = STATE["paused"]
            display_off = STATE["display_off"]
        if paused:
            if not placeholder_once:
                placeholder("Detection paused")
                placeholder_once = True
            time.sleep(0.2)
            continue
        if display_off:
            if not placeholder_once:
                placeholder("Camera view off - detection continues")
                placeholder_once = True
            time.sleep(0.2)
            continue
        placeholder_once = False
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        vis, dets, ms = infer_one(frame)
        push_view(vis)
        update_stats(len(dets), ms)
        apply_reject(len(dets))
    cap.release()
    with STATE["lock"]:
        STATE["mode"] = "idle"
    log("摄像头已关闭并释放")


def run_images(paths, session_dir):
    with STATE["lock"]:
        STATE["mode"] = "images"
        STATE["records"] = []          # 新会话：清空检测记录表
        STATE["reject_cnt"] = 0        # 新会话：剔除计数归零
    all_dir = os.path.join(session_dir, "all")
    os.makedirs(all_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "result.csv")
    total = len(paths)
    log("开始图片批量检测，共 %d 张..." % total)
    done = 0
    for p in paths:
        if stop_requested():
            log("检测已停止")
            return
        while STATE["paused"]:
            if stop_requested():
                log("检测已停止")
                return
            time.sleep(0.2)
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log("[错误] 图片无法读取，已跳过：" + os.path.basename(p))
            continue
        vis, dets, ms = infer_one(img)
        count = len(dets)
        name = os.path.splitext(os.path.basename(p))[0] + ".jpg"
        cv2.imwrite(os.path.join(all_dir, name), vis)
        record_row(os.path.basename(p), "图片", count, ms, csv_path)
        push_view(vis)
        update_stats(count, ms)
        apply_reject(count, per_item=True)   # 图片模式：每张有毛发的图都计数
        done += 1
        log("[%d/%d] %s：检出 %d 根，%.0f ms" % (done, total, os.path.basename(p), count, ms))
    # 不覆盖画面：左侧保留最后一张检测结果图
    with STATE["lock"]:
        STATE["mode"] = "idle"       # 批量结束自动回到空闲，按钮立即可用
    log("图片检测完成（共 %d 张），结果保存在 %s" % (done, os.path.basename(session_dir)))


def run_video(path, session_dir):
    with STATE["lock"]:
        STATE["mode"] = "video"
        STATE["records"] = []          # 新会话：清空检测记录表
        STATE["reject_cnt"] = 0        # 新会话：剔除计数归零
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log("[错误] 视频无法打开：" + os.path.basename(path))
        with STATE["lock"]:
            STATE["mode"] = "idle"
        return
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = "video_" + time.strftime("%H%M%S") + ".mp4"
    out_path = os.path.join(session_dir, name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    csv_path = os.path.join(session_dir, "result.csv")
    log("开始视频检测：%s（%dx%d）" % (os.path.basename(path), w, h))
    frames = 0
    max_count = 0
    ms_sum = 0.0
    t0 = time.time()
    while True:
        if stop_requested():
            break
        while STATE["paused"]:
            if stop_requested():
                break
            time.sleep(0.2)
        if stop_requested():
            break
        ok, frame = cap.read()
        if not ok:
            break
        vis, dets, ms = infer_one(frame)
        writer.write(vis)
        push_view(vis)
        max_count = max(max_count, len(dets))
        ms_sum += ms
        update_stats(len(dets), ms)
        apply_reject(len(dets))
        frames += 1
    cap.release()
    writer.release()
    used = time.time() - t0
    avg_ms = ms_sum / frames if frames else 0
    record_row(os.path.basename(path), "视频", max_count, avg_ms, csv_path)
    log("视频检测完成：%d 帧，最大同时检出 %d 根，用时 %.0f 秒，已保存 %s"
        % (frames, max_count, used, name))
    # mp4v 编码浏览器无法播放，自动转码 H.264 供网页播放（装有 ffmpeg 时）
    try:
        if os.path.exists("/usr/bin/ffmpeg"):
            play_path = out_path.replace(".mp4", "_h264.mp4")
            rc = os.system('ffmpeg -y -i "%s" -vcodec libx264 -pix_fmt yuv420p "%s" > /dev/null 2>&1'
                           % (out_path, play_path))
            if rc == 0 and os.path.exists(play_path) and os.path.getsize(play_path) > 1000:
                log("已生成网页可播放版本：" + os.path.basename(play_path))
            else:
                log("[提示] H.264 转码未成功，视频请下载后播放")
    except Exception as exc:
        log("[提示] 转码出错：%s（视频请下载后播放）" % exc)
    # 不覆盖画面：左侧保留最后一帧检测结果
    with STATE["lock"]:
        STATE["mode"] = "idle"


# ---------- 后台工作线程 ----------
def worker():
    load_net()
    with STATE["lock"]:
        STATE["net_ready"] = True
    log("模型加载完成：%s（FP16 NCNN，imgsz=1280）" % os.path.basename(MODEL_DIR))
    placeholder("Waiting for detection...")
    while True:
        action = pop_action()
        if action is None:
            time.sleep(0.1)
            continue
        kind = action[0]
        if kind == "stop":
            with STATE["lock"]:
                STATE["mode"] = "idle"
                STATE["paused"] = False
            log("检测已停止")
            placeholder("Detection stopped")
        elif kind == "camera":
            run_camera(action[1])
        elif kind == "images":
            run_images(action[1], action[2])
        elif kind == "video":
            run_video(action[1], action[2])
        apply_reject(0)   # 每轮会话结束：剔除状态复位（圆灯/板载灯熄灭）


def temp_loop():
    while True:
        try:
            t = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
            STATE["temp"] = "%.1f °C" % (int(t) / 1000.0)
        except Exception:
            STATE["temp"] = "--"
        time.sleep(5)


# ---------- MJPEG 视图流 ----------
def jpeg_stream():
    last = -1
    while True:
        with STATE["lock"]:
            fid = STATE["view_id"]
            f = STATE["view"]
        if f is None or fid == last:
            time.sleep(0.02)
            continue
        last = fid
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
               str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")


# ---------- 网页 ----------
PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>毛发异物检测 · 树莓派控制台</title>
<style>
body{margin:0;background:#f5f5f5;color:#333;font-family:system-ui,'Microsoft YaHei',sans-serif;font-size:14px}
.top{display:flex;gap:22px;flex-wrap:wrap;align-items:center;padding:10px 16px;background:#fff;border-bottom:1px solid #e2e2e2}
.top b{color:#185fa5;font-weight:500}
.top a{color:#185fa5;text-decoration:none;font-size:13px}
.wrap{display:flex;height:calc(100vh - 53px)}
.left{flex:1;display:flex;min-width:0;background:#111}
.left img{display:block;width:100%;height:100%;object-fit:contain}
.right{width:380px;border-left:1px solid #e2e2e2;overflow-y:auto;padding:10px;background:#f5f5f5}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:10px;margin-bottom:10px}
.card h4{margin:2px 0 8px;color:#185fa5;font-weight:500;font-size:13px}
.tabs{display:flex;background:#efefef;border:1px solid #ddd;border-radius:7px;padding:3px;gap:3px;margin-bottom:8px}
.tabs button{flex:1;background:transparent;color:#888;border:none;border-radius:5px;padding:6px 0;font-size:12px;cursor:pointer}
.tabs button.on{background:#fff;color:#185fa5;border:1px solid #c8dcf4;font-weight:500}
.zone{border:1.5px dashed #bbb;border-radius:8px;padding:20px 10px;text-align:center;color:#888;font-size:12px;cursor:pointer;background:#fafafa}
.zone:hover{border-color:#185fa5}
.zone .ic{width:34px;height:34px;margin:0 auto 8px;border:2px solid #bbb;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#999;font-size:20px}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
button{background:#f2f2f2;color:#333;border:1px solid #ddd;border-radius:6px;padding:8px 0;font-size:12px;cursor:pointer}
button:hover{background:#e8e8e8}
button.go{width:100%;background:#3b6d11;color:#fff;border:1px solid #27500a;border-radius:7px;padding:9px 0;font-size:13px;font-weight:500;margin-top:10px}
button:disabled{opacity:.45;cursor:not-allowed}
select,input[type=file]{background:#fafafa;color:#333;border:1px solid #ddd;border-radius:6px;padding:6px 8px;font-size:12px}
input[type=file]{width:100%}
input[type=range]{width:100%}
.log{height:150px;overflow-y:auto;background:#fafafa;border:1px solid #e2e2e2;border-radius:6px;padding:6px;font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap;color:#3a6b35}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{border:1px solid #e2e2e2;padding:3px 6px;text-align:center}
th{background:#f2f2f2;color:#185fa5}
.msg{color:#3b6d11}
</style></head><body>
<div class="top">
<span>毛发数量：<b id="cnt">0</b></span>
<span>单帧耗时：<b id="ms">0 ms</b></span>
<span>帧率：<b id="fps">0.0</b></span>
<span>CPU 温度：<b id="temp">--</b></span>
<span>模式：<b id="mode">空闲</b></span>
</div>
<div class="wrap">
<div class="left"><img src="/stream" alt="检测结果"></div>
<div class="right">
  <div class="card">
    <h4>检测源</h4>
    <div class="tabs">
      <button id="tabFile" class="on" onclick="pick('file')">文件</button>
      <button id="tabDir" onclick="pick('dir')">文件夹</button>
      <button id="tabCam" onclick="pick('cam')">摄像头</button>
    </div>
    <div class="zone" id="zone" onclick="pickFile()">
      <div class="ic">+</div>
      <div id="zoneMain">点击选择图片或视频（可多选）</div>
      <div id="zoneSub">上传后自动识别类型：图片批量检测 / 视频逐个检测</div>
    </div>
    <input type="file" id="fPick" multiple accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.mp4,.avi,.mov,.mkv" style="display:none">
    <input type="file" id="fDir" webkitdirectory multiple style="display:none" onchange="dirChosen(this)">
    <div id="camRow" style="display:none;margin:4px 0 2px">
      <div style="color:#999;font-size:12px;margin-bottom:4px">摄像头设备</div>
      <select id="camSel" style="width:100%"><option>摄像头 0</option></select>
    </div>
    <button class="go" id="goBtn" onclick="startAction()">▶ 开始检测</button>
  </div>
  <div class="card">
    <h4>运行控制</h4>
    <div class="grid2">
      <button onclick="togglePause()">暂停</button>
      <button onclick="stopAll()" style="color:#a32d2d">停止检测</button>
      <button onclick="toggleDisplay()">关闭画面</button>
      <button onclick="window.open('/results','_blank')">打开输出目录</button>
    </div>
  </div>
  <div class="card">
    <h4 style="display:flex;justify-content:space-between;align-items:center;margin:2px 0 10px">
      <span>剔除模拟</span>
      <label style="display:flex;align-items:center;gap:5px;font-size:12px;color:#666;font-weight:400;cursor:pointer">
        <input type="checkbox" id="rejEn" onchange="setRejectEn()"> 启用
      </label>
    </h4>
    <div style="display:flex;gap:14px;align-items:center;margin-bottom:12px">
      <div id="valve" style="width:56px;height:56px;border-radius:50%;background:#e6e6e6;border:2px solid #ccc"></div>
      <div style="flex:1">
        <div style="font-size:12px;color:#777;margin-bottom:4px">剔除动作状态</div>
        <div id="rejState" style="font-size:14px;font-weight:500;color:#3b6d11">未检出毛发</div>
        <div style="font-size:12px;color:#999;margin-top:4px">板载 LED：<span id="ledTxt" style="color:#888">熄灭</span></div>
      </div>
    </div>
  </div>
  <div class="card">
    <h4>判定参数</h4>
    <div style="font-size:12px;color:#777;margin-bottom:2px">置信度（即时生效）：<b id="confv" style="color:#222">0.45</b></div>
    <input type="range" id="conf" min="0.01" max="0.99" step="0.01" value="0.45" oninput="setConf(this.value)">
    <div style="font-size:12px;color:#777;margin:8px 0 4px">推理尺寸（下一帧生效）</div>
    <select id="imgsz" style="width:100%" onchange="setImgsz(this.value)">
      <option value="640">640p</option><option value="960">960p</option>
      <option value="1280" selected>1080p（1280）</option><option value="1920">2K</option>
      <option value="2560">4K</option>
    </select>
  </div>
  <div class="card">
    <h4>运行日志</h4>
    <div class="log" id="log"></div>
  </div>
  <div class="card">
    <h4>检测记录</h4>
    <table><thead><tr><th>文件</th><th>类型</th><th>毛发数量</th><th>耗时ms</th></tr></thead>
    <tbody id="recBody"></tbody></table>
  </div>
</div>
</div>
<script>
let tab = 'file';
let dirFiles = [];
function dirChosen(inp){
  dirFiles = Array.from(inp.files).filter(f =>
    /\.(jpe?g|png|bmp|tiff?|mp4|avi|mov|mkv)$/i.test(f.name));
  document.getElementById('zoneMain').textContent = dirFiles.length ?
    ('已选 ' + dirFiles.length + ' 个文件，点击「开始检测」提交') : '文件夹内没有可识别的图片或视频';
}
const msgEl = document.getElementById('msg');
function msg(t){ const m=document.getElementById('msg'); }
function flash(t){ const d=document.createElement('div'); d.className='msg'; d.textContent=t;
  d.style.cssText='position:fixed;top:60px;right:20px;background:#fff;border:1px solid #3b6d11;color:#3b6d11;border-radius:6px;padding:8px 14px;z-index:9';
  document.body.appendChild(d); setTimeout(()=>d.remove(),3000); }
function pick(k){
  tab = k;
  document.getElementById('tabFile').className = k==='file' ? 'on' : '';
  document.getElementById('tabDir').className = k==='dir' ? 'on' : '';
  document.getElementById('tabCam').className = k==='cam' ? 'on' : '';
  document.getElementById('zone').style.display = k==='cam' ? 'none' : 'block';
  document.getElementById('camRow').style.display = k==='cam' ? 'block' : 'none';
  document.getElementById('goBtn').textContent = k==='cam' ? '▶ 打开摄像头' : '▶ 开始检测';
  const zm = document.getElementById('zoneMain'), zs = document.getElementById('zoneSub');
  if(k === 'file'){ zm.textContent = '点击选择图片或视频（可多选）'; zs.textContent = '上传后自动识别类型：图片批量检测 / 视频逐个检测'; }
  if(k === 'dir'){ zm.textContent = '点击选择文件夹'; zs.textContent = '自动识别其中的图片和视频'; }
}
function pickFile(){ (tab === 'dir' ? document.getElementById('fDir') : document.getElementById('fPick')).click(); }
function chosen(inp){
  document.getElementById('zoneMain').textContent = '已选择 ' + inp.files.length + ' 个文件，点击「开始检测」提交';
}
async function startAction(){
  if(tab === 'cam'){ await startCamera(); return; }
  let fd = new FormData(), n = 0, label = '';
  if(dirFiles.length){
    dirFiles.forEach(f => fd.append('files', f));
    n = dirFiles.length; label = n + ' 个文件（文件夹）';
  } else {
    const fs = document.getElementById('fPick').files;
    if(!fs.length){ flash('请先选择文件夹或文件'); return; }
    for(const f of fs) fd.append('files', f);
    n = fs.length; label = n + ' 个文件';
  }
  flash('上传中（共 ' + n + ' 个）...');
  const r = await (await fetch('/api/upload_files', {method:'POST', body:fd})).json();
  flash(r.ok ? ('已提交：' + r.images + ' 张图片 + ' + r.videos + ' 个视频（' + label + '）') : '上传失败');
  if(r.ok){ document.getElementById('fPick').value = ''; document.getElementById('fDir').value = '';
    dirFiles = []; document.getElementById('zoneMain').textContent = '点击选择图片或视频（可多选）'; }
}
async function startCamera(){
  const dev = document.getElementById('camSel').value;
  if(dev === ''){ flash('未检测到摄像头'); return; }
  await fetch('/api/camera', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'dev='+dev});
  flash('摄像头检测已启动');
}
async function togglePause(){
  const r = await (await fetch('/api/pause', {method:'POST'})).json();
  document.querySelector('[onclick="togglePause()"]').textContent = r.paused ? '继续' : '暂停';
  flash(r.paused ? '已暂停' : '已继续');
}
async function stopAll(){ await fetch('/api/stop', {method:'POST'}); flash('已停止'); }
async function toggleDisplay(){
  const r = await (await fetch('/api/toggle_display', {method:'POST'})).json();
  flash(r.display_off ? '画面已关闭（检测继续）' : '画面已恢复');
}
async function setConf(v){
  document.getElementById('confv').textContent = Number(v).toFixed(2);
  await fetch('/api/conf', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'value='+v});
}
async function setImgsz(v){
  const r = await (await fetch('/api/imgsz', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'value='+v})).json();
  flash(r.ok ? ('推理尺寸已切换 ' + v) : r.msg);
  document.getElementById('imgsz').value = r.imgsz;
}
async function setRejectEn(){
  const en = document.getElementById('rejEn').checked;
  const r = await (await fetch('/api/reject_enable',{method:'POST'})).json();
  flash(r.enabled ? '剔除模拟已启用' : '剔除模拟已停用（纯检测）');
}
async function refresh(){
  const d = await (await fetch('/api/stats')).json();
  document.getElementById('cnt').textContent = d.count;
  document.getElementById('ms').textContent = d.ms.toFixed(0) + ' ms';
  document.getElementById('fps').textContent = d.fps.toFixed(1) + ' 帧/秒';
  document.getElementById('temp').textContent = d.temp;
  document.getElementById('mode').textContent = d.mode_text;
  document.getElementById('confv').textContent = d.conf.toFixed(2);
  document.getElementById('log').textContent = d.log.join('\\n');
  document.getElementById('log').scrollTop = 1e9;
  const tb = document.getElementById('recBody'); tb.innerHTML='';
  d.records.slice().reverse().forEach(r=>{
    const tr=document.createElement('tr');
    r.forEach(c=>{const td=document.createElement('td');td.textContent=c;tr.appendChild(td);});
    tb.appendChild(tr);
  });
  document.querySelector('[onclick="togglePause()"]').textContent = d.paused ? '继续' : '暂停';
  const rejActive = d.reject_en && d.reject_active;
  const vv = document.getElementById('valve');
  vv.style.background = rejActive ? '#e24b4a' : '#e6e6e6';
  vv.style.borderColor = rejActive ? '#a32d2d' : '#ccc';
  document.getElementById('rejState').textContent = rejActive ? '检出毛发' : '未检出毛发';
  document.getElementById('rejState').style.color = rejActive ? '#a32d2d' : '#3b6d11';
  document.getElementById('ledTxt').textContent = rejActive ? '常亮（绿色）' : '熄灭';
  document.getElementById('ledTxt').style.color = rejActive ? '#3b6d11' : '#888';
  document.getElementById('rejEn').checked = d.reject_en;
  const cs = document.getElementById('camSel');
  if(d.cameras.join() !== cs.dataset.list){
    cs.dataset.list = d.cameras.join(); cs.innerHTML='';
    if(!d.cameras.length){ const o=document.createElement('option'); o.value=''; o.textContent='未检测到摄像头'; cs.appendChild(o); }
    d.cameras.forEach(c=>{const o=document.createElement('option'); o.value=c; o.textContent='摄像头 '+c; cs.appendChild(o);});
  }
}
document.getElementById('fPick').addEventListener('change', function(){ chosen(this); });
setInterval(refresh, 1500); refresh();
</script></body></html>"""

RESULTS_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>检测结果与报表</title>
<style>
body{background:#f5f5f5;color:#333;font-family:system-ui,'Microsoft YaHei',sans-serif;padding:20px;font-size:14px}
a{color:#185fa5;text-decoration:none}
h3,h4{color:#185fa5}
.sess{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:14px 16px;margin-bottom:20px}
.sess h4{margin:0 0 10px}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.cell{background:#fafafa;border:1px solid #e2e2e2;border-radius:8px;padding:8px;width:230px}
.cell img{width:100%;height:150px;object-fit:cover;display:block;border-radius:4px;background:#111}
.cell video{width:100%;border-radius:4px;display:block}
.cap{font-size:12px;color:#888;margin-top:4px;word-break:break-all}
.tip{font-size:12px;color:#b26a00;background:#fff7e6;border:1px solid #f0dcb0;border-radius:4px;padding:16px 8px;text-align:center}
table{border-collapse:collapse;font-size:13px;margin:6px 0}
th,td{border:1px solid #e2e2e2;padding:4px 10px;text-align:center}
th{background:#f2f2f2;color:#185fa5}
.dl{display:inline-block;background:#3b6d11;border:1px solid #27500a;color:#fff;border-radius:6px;padding:4px 12px;margin:4px 0}
.empty{color:#999}
</style></head><body>
<h3><a href="javascript:void 0" onclick="closeBack()">← 返回控制台</a> ｜ 检测结果与报表</h3>
<script>
function closeBack(){
  // 结果页由控制台 window.open 打开，可直接关闭；直接输入网址打开时关闭会失败，回退为跳转
  window.close();
  setTimeout(function(){ location.href = '/'; }, 200);
}
</script>
{% for s in sessions %}
<div class="sess">
<h4>{{ s.name }}　<a class="dl" href="/results/zip?session={{ s.name }}">⬇ 下载 ZIP</a></h4>
{% if s['images'] %}
<div class="grid">
{% for f in s['images'] %}<div class="cell">
  <a href="/results/file?path={{ f.rel }}" target="_blank"><img src="/results/file?path={{ f.rel }}" alt="{{ f.rel }}"></a>
  <div class="cap">{{ f.name }}<br>{{ f.size }} · <a href="/results/download?path={{ f.rel }}">下载</a></div>
</div>{% endfor %}
</div>
{% else %}<div class="empty">本轮无图片检测</div>{% endif %}
{% if s['videos'] %}
{% for f in s['videos'] %}<div class="cell">
  {% if '_h264' in f.rel %}<video controls preload="metadata" src="/results/file?path={{ f.rel }}"></video>
  {% else %}<div class="tip">mp4v 编码浏览器不支持在线播放<br>请下载后播放，或查看 _h264 版本</div>{% endif %}
  <div class="cap">{{ f.name }}<br>{{ f.size }} · <a href="/results/download?path={{ f.rel }}">下载</a></div>
</div>{% endfor %}
{% else %}<div class="empty">本轮无视频检测</div>{% endif %}
{% if s['csv'] %}
{% for c in s['csv'] %}
<div class="cap">{{ c.name }}　<a class="dl" href="/results/download?path={{ c.rel }}">下载 CSV</a></div>
<table><tr>{% for h in c.rows[0] %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in c.rows[1:] %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</table>
{% endfor %}
{% else %}<div class="empty">本轮无报表</div>{% endif %}
</div>
{% endfor %}{% if not sessions %}<div class="empty">暂无检测记录</div>{% endif %}
{% if legacy %}
<h4>历史文件（legacy）</h4>
{% for f in legacy %}<div>{{ f.rel }} · {{ f.size }} · <a href="/results/download?path={{ f.rel }}">下载</a></div>{% endfor %}
{% endif %}
</body></html>"""


def classify_files(rel_root):
    # 分类某个目录下的文件（rel 相对 results\ 根目录）
    cats = {"images": [], "videos": [], "csv": [], "others": []}
    for root, dirs, files in os.walk(os.path.join(RESULT_DIR, rel_root)):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, RESULT_DIR).replace(os.sep, "/")
            st = os.stat(full)
            item = {"rel": rel, "name": name, "size": "%.1f KB" % (st.st_size / 1024)}
            ext = os.path.splitext(name)[1].lower()
            if ext in (".jpg", ".jpeg", ".png"):
                cats["images"].append(item)
            elif ext in (".mp4", ".avi", ".mov", ".mkv"):
                cats["videos"].append(item)
            elif ext == ".csv":
                cats["csv"].append(item)
            else:
                cats["others"].append(item)
    for key in cats:
        cats[key].sort(key=lambda x: x["rel"])
    return cats


def list_sessions():
    # 每个时间戳目录 = 一轮检测；其他文件归入历史（legacy）
    import re as _re
    sessions = []
    legacy = []
    if not os.path.isdir(RESULT_DIR):
        return sessions, legacy
    for entry in sorted(os.listdir(RESULT_DIR), reverse=True):
        full = os.path.join(RESULT_DIR, entry)
        if os.path.isdir(full) and _re.match(r"^\d{8}_\d{6}(_\d{2})?$", entry):
            sess = classify_files(entry)
            sess["name"] = entry
            if sess["images"] or sess["videos"] or sess["csv"] or sess["others"]:
                sessions.append(sess)
        else:
            if os.path.isdir(full):
                cats = classify_files(entry)
                for key in ("images", "videos", "csv", "others"):
                    for item in cats[key]:
                        legacy.append(item)
            elif os.path.isfile(full):
                st = os.stat(full)
                legacy.append({"rel": entry, "name": entry,
                               "size": "%.1f KB" % (st.st_size / 1024)})
    return sessions, legacy


@app.route("/")
def index():
    return PAGE


@app.route("/stream")
def stream():
    return Response(jpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def stats():
    with STATE["lock"]:
        mode_text = {"idle": "空闲", "camera": "摄像头实时", "images": "图片批量",
                     "video": "视频检测"}[STATE["mode"]]
        return jsonify({
            "count": STATE["count"], "ms": STATE["ms"], "fps": STATE["fps"],
            "temp": STATE["temp"], "conf": STATE["conf"], "imgsz": CONFIG["imgsz"],
            "mode": STATE["mode"], "mode_text": mode_text,
            "paused": STATE["paused"], "display_off": STATE["display_off"],
            "net_ready": STATE["net_ready"],
            "reject_en": STATE["reject_en"],
            "reject_cnt": STATE["reject_cnt"],
            "reject_active": STATE["reject_active"],
            "cameras": list_cameras(),
            "log": STATE["log"][-40:],
            "records": STATE["records"][-100:],
        })


def list_cameras():
    # 树莓派会在 /dev/video* 注册大量内部节点（rpi-hevc / pispbe 等），
    # 按 /sys 中的设备名过滤，只保留真实摄像头
    internal = ["rpi-hevc", "pispbe", "unicam", "rp1-cfe", "rpivid"]
    devs = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*"),
                       key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0)):
        try:
            name = open(os.path.join(path, "name")).read().strip().lower()
        except OSError:
            name = ""
        if any(p in name for p in internal):
            continue
        digits = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
        try:
            devs.append(int(digits))
        except ValueError:
            pass
    return devs


@app.route("/api/cameras")
def cameras():
    return jsonify({"cameras": list_cameras()})


@app.route("/api/conf", methods=["POST"])
def set_conf():
    try:
        v = min(0.99, max(0.01, float(request.form.get("value", 0.45))))
    except ValueError:
        return jsonify({"ok": False})
    with STATE["lock"]:
        STATE["conf"] = v
    log("置信度调整为 %.2f（即时生效）" % v)
    return jsonify({"ok": True, "conf": v})


@app.route("/api/imgsz", methods=["POST"])
def set_imgsz():
    try:
        v = int(request.form.get("value", 1280))
    except ValueError:
        return jsonify({"ok": False, "msg": "尺寸无效", "imgsz": CONFIG["imgsz"]})
    old = CONFIG["imgsz"]
    CONFIG["imgsz"] = v
    with STATE["lock"]:
        STATE["imgsz"] = v
    if v != old:
        log("推理尺寸调整为 %d（下一帧生效）" % v)
    return jsonify({"ok": True, "imgsz": v})


@app.route("/api/camera", methods=["POST"])
def api_camera():
    try:
        dev = int(request.form.get("dev", 0))
    except ValueError:
        return jsonify({"ok": False}), 400
    with STATE["lock"]:
        STATE["camera_dev"] = dev
        STATE["paused"] = False
    with ACTION_LOCK:
        ACTIONS.append(("camera", dev))
    return jsonify({"ok": True})


@app.route("/api/toggle_display", methods=["POST"])
def api_toggle_display():
    with STATE["lock"]:
        STATE["display_off"] = not STATE["display_off"]
    return jsonify({"display_off": STATE["display_off"]})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    with STATE["lock"]:
        STATE["paused"] = not STATE["paused"]
        paused = STATE["paused"]
    log("已暂停" if paused else "已继续")
    return jsonify({"paused": paused})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with ACTION_LOCK:
        ACTIONS.clear()            # 停止 = 中止当前 + 清空所有排队任务
        ACTIONS.append(("stop",))
    return jsonify({"ok": True})


@app.route("/api/upload_files", methods=["POST"])
def api_upload_files():
    # 统一上传入口：按扩展名自动分流（图片进批量检测，视频逐个排队）
    # 每次提交 = 一轮检测，独立创建 results/<时间戳>/ 会话目录
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False}), 400
    os.makedirs(RESULT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(RESULT_DIR, ts)
    n = 1
    while os.path.exists(session_dir):
        n += 1
        session_dir = os.path.join(RESULT_DIR, "%s_%02d" % (ts, n))
    os.makedirs(session_dir, exist_ok=True)
    images, videos = [], []
    for i, f in enumerate(files):
        ext = os.path.splitext(f.filename)[1].lower()
        name = "%s_%02d_%s" % (ts, i, os.path.basename(f.filename))
        path = os.path.join(UPLOAD_DIR, name)
        f.save(path)
        if ext in IMAGE_EXTS:
            images.append(path)
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            videos.append(path)
    with STATE["lock"]:
        STATE["paused"] = False
    with ACTION_LOCK:
        if images:
            ACTIONS.append(("images", images, session_dir))
        for v in videos:
            ACTIONS.append(("video", v, session_dir))
    log("第 %s 轮检测已提交：图片 %d 张、视频 %d 个（按顺序执行）"
        % (os.path.basename(session_dir), len(images), len(videos)))
    return jsonify({"ok": True, "images": len(images), "videos": len(videos),
                    "count": len(files), "session": os.path.basename(session_dir)})


@app.route("/api/reject_enable", methods=["POST"])
def api_reject_enable():
    with STATE["lock"]:
        STATE["reject_en"] = not STATE["reject_en"]
        en = STATE["reject_en"]
        if not en:
            STATE["last_hair"] = False
            STATE["reject_active"] = False
    if not en:
        act_led(False)
    log("剔除模拟已" + ("启用" if en else "停用"))
    return jsonify({"enabled": en})


@app.route("/results")
def results_page():
    sessions, legacy = list_sessions()
    csv_rows = []
    for s in sessions:
        for c in s["csv"]:
            full = os.path.join(RESULT_DIR, c["rel"])
            try:
                with open(full, newline="", encoding="utf-8-sig") as fp:
                    rows = list(csv.reader(fp))
                c["rows"] = rows[:200]
            except Exception:
                c["rows"] = [["（读取失败）"]]
    return render_template_string(RESULTS_PAGE, sessions=sessions,
                                  csv_rows=csv_rows, legacy=legacy)


@app.route("/results/file")
def results_file():
    # 内联显示（图片预览 / 视频播放）
    rel = request.args.get("path", "")
    safe = os.path.normpath(rel).lstrip("/\\")
    full = os.path.join(RESULT_DIR, safe)
    if not os.path.isfile(full):
        return "文件不存在", 404
    return send_from_directory(RESULT_DIR, safe)


@app.route("/results/download")
def results_download():
    rel = request.args.get("path", "")
    safe = os.path.normpath(rel).lstrip("/\\")
    full = os.path.join(RESULT_DIR, safe)
    if not os.path.isfile(full):
        return "文件不存在", 404
    return send_from_directory(RESULT_DIR, safe, as_attachment=True)


@app.route("/results/zip")
def results_zip():
    # 整轮打包下载：把一个会话目录压缩为 <轮次>.zip
    name = request.args.get("session", "")
    if not re.match(r"^\d{8}_\d{6}(_\d{2})?$", name):
        return "轮次无效", 400
    sess_dir = os.path.join(RESULT_DIR, name)
    if not os.path.isdir(sess_dir):
        return "轮次不存在", 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(sess_dir):
            for f in sorted(files):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, sess_dir).replace(os.sep, "/")
                zf.write(full, name + "/" + rel)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=name + ".zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    CONFIG["port"] = a.port
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=temp_loop, daemon=True).start()
    print("控制台已启动: http://0.0.0.0:%d" % a.port, flush=True)
    app.run(host="0.0.0.0", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
