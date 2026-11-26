# -*- coding: utf-8 -*-
"""
树莓派5 毛发检测模型验证脚本（纯 ncnn + numpy，不依赖 ultralytics）
推理代码与网页控制台(web_console.py)完全一致：letterbox/NMS/4线程/1280尺寸
四项测试：A 边缘精度 / B 推理速度 / C 阈值扫描 / D 背景误检
输出：~/yolo/eval/ 下的 results.md + det_val.json + det_bg.json + speed.json
"""
import os, json, time, glob
import numpy as np
import cv2
import ncnn

BASE   = os.path.expanduser("~/yolo")
MODEL_DIR = os.path.join(BASE, "hair_yolov8n_v3_20261016_fp16_ncnn_model")
VAL_IMG   = os.path.join(BASE, "val_images")
VAL_LBL   = os.path.join(BASE, "val_labels")
BG_DIR    = os.path.join(BASE, "bg_test")
OUT       = os.path.join(BASE, "eval")
IMGSZ     = 1280
NMS_THR   = 0.45
DUMP_CONF = 0.01          # 低阈值推理一次，所有阈值离线算
THRESHOLDS = [round(0.05*i, 2) for i in range(1, 19)]   # 0.05 ~ 0.90
IOU_THR   = 0.5

os.makedirs(OUT, exist_ok=True)

# ---------- 推理核心（照抄 web_console.py） ----------
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

def nms(bxy, sc, thr=NMS_THR):
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

def detect(img, conf, imgsz=IMGSZ):
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
        dets = [(list(map(float, b)), float(s)) for b, s in zip(bxy, sc2)]
    return dets

def imread(path):
    return cv2.imread(path)

def list_images(root):
    exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(exts):
                out.append(os.path.join(dirpath, f))
    return sorted(out)

# ---------- 测试 B：推理速度（100帧计时） ----------
def test_speed(sample_img):
    for _ in range(3):
        detect(sample_img, 0.25)          # 预热3帧
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        detect(sample_img, 0.25)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    res = {"mean_ms": float(ts.mean()), "std_ms": float(ts.std()),
           "min_ms": float(ts.min()), "max_ms": float(ts.max()),
           "fps": float(1000.0 / ts.mean()), "frames": 100}
    print("[B] 速度: 平均 %.1f ms / 帧 (std %.1f), %.2f FPS" % (res["mean_ms"], res["std_ms"], res["fps"]))
    return res

# ---------- 推理并转存检出 JSON ----------
def dump_dir(img_root, json_path, conf=DUMP_CONF):
    files = list_images(img_root)
    out = {}
    t_start = time.time()
    for i, p in enumerate(files, 1):
        img = imread(p)
        if img is None:
            print("[跳过] 无法读取:", p); continue
        dets = detect(img, conf)
        out[os.path.relpath(p, img_root)] = {
            "boxes": [[round(v, 2) for v in b] for b, _ in dets],
            "scores": [round(float(s), 4) for _, s in dets],
            "w": img.shape[1], "h": img.shape[0]}
        if i % 50 == 0:
            print("  %d/%d  已用 %.0f s" % (i, len(files), time.time() - t_start))
    with open(json_path, "w") as f:
        json.dump(out, f)
    print("[dump] %s: %d 张, 耗时 %.0f s" % (img_root, len(out), time.time() - t_start))
    return out

# ---------- 指标计算（纯 numpy） ----------
def iou_xyxy(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)

def load_labels(lbl_dir, name, w, h):
    p = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
    boxes = []
    if os.path.exists(p):
        for line in open(p):
            t = line.split()
            if len(t) >= 5:
                cx, cy, bw, bh = (float(t[1])*w, float(t[2])*h, float(t[3])*w, float(t[4])*h)
                boxes.append([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])
    return boxes

def eval_val(det_json, lbl_dir):
    """返回：各阈值 P/R/F1 + 全局 AP50"""
    data = json.load(open(det_json))
    names = sorted(data.keys())
    gts = {n: load_labels(lbl_dir, n, data[n]["w"], data[n]["h"]) for n in names}
    total_gt = sum(len(g) for g in gts.values())

    # --- AP50：全部检出按分数排序做全局匹配 ---
    all_preds = []
    for n in names:
        for b, s in zip(data[n]["boxes"], data[n]["scores"]):
            all_preds.append((s, n, b))
    all_preds.sort(key=lambda x: -x[0])
    matched = {n: [False]*len(gts[n]) for n in names}
    tp, fp = 0, 0
    pr_curve = []
    for s, n, b in all_preds:
        best_iou, best_j = IOU_THR, -1
        for j, g in enumerate(gts[n]):
            if matched[n][j]:
                continue
            v = iou_xyxy(b, g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched[n][best_j] = True
            tp += 1
        else:
            fp += 1
        pr_curve.append((tp / (tp+fp), tp / total_gt))
    # 全点插值 AP
    p_env, ap = 0.0, 0.0
    for p, r in reversed(pr_curve):
        p_env = max(p_env, p)
        ap += p_env * 0  # 占位，下面用标准法
    # 标准全点插值：单调递减包络
    rs = [r for _, r in pr_curve]
    ps = [p for p, _ in pr_curve]
    env = []
    m = 0.0
    for p in reversed(ps):
        m = max(m, p)
        env.append(m)
    env.reverse()
    ap = 0.0
    prev_r = 0.0
    for (p, r), pe in zip(pr_curve, env):
        ap += pe * (r - prev_r)
        prev_r = r
    ap50 = ap

    # --- 各阈值 P/R/F1 ---
    rows = []
    best_f1, best_thr = 0.0, 0.0
    for thr in THRESHOLDS:
        TP = FP = FN = 0
        for n in names:
            g = gts[n]
            preds = [(b, s) for b, s in zip(data[n]["boxes"], data[n]["scores"]) if s >= thr]
            preds.sort(key=lambda x: -x[1])
            used = [False]*len(g)
            for b, s in preds:
                best_iou, best_j = IOU_THR, -1
                for j, gg in enumerate(g):
                    if used[j]:
                        continue
                    v = iou_xyxy(b, gg)
                    if v >= best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0:
                    used[best_j] = True
                    TP += 1
                else:
                    FP += 1
            FN += len(g) - sum(used)
        P = TP / (TP+FP) if TP+FP else 0.0
        R = TP / (TP+FN) if TP+FN else 0.0
        F1 = 2*P*R/(P+R) if P+R else 0.0
        rows.append((thr, P, R, F1, TP, FP, FN))
        if F1 > best_f1:
            best_f1, best_thr = F1, thr
    print("[A/C] 验证集 %d 张, GT %d 个: AP50=%.4f, F1最优阈值=%.2f (F1=%.4f)" % (len(names), total_gt, ap50, best_thr, best_f1))
    return {"n_images": len(names), "total_gt": total_gt, "ap50": ap50,
            "rows": rows, "best_thr": best_thr, "best_f1": best_f1}

def eval_bg(det_json):
    """背景误检：各阈值下出现误报的图片数与误报框数"""
    data = json.load(open(det_json))
    rows = []
    for thr in THRESHOLDS:
        n_img = 0
        n_box = 0
        for n, d in data.items():
            k = sum(1 for s in d["scores"] if s >= thr)
            if k:
                n_img += 1
                n_box += k
        rows.append((thr, n_img, n_box))
    print("[D] 背景 %d 张: 阈值0.25时误报图 %d 张 / 误报框 %d 个" % (
        len(data), dict((r[0], r[1]) for r in rows).get(0.25, -1),
        dict((r[0], r[2]) for r in rows).get(0.25, -1)))
    return {"n_images": len(data), "rows": rows}

# ---------- 主流程 ----------
if __name__ == "__main__":
    print("=== 加载模型 ===")
    load_net()
    val_files = list_images(VAL_IMG)
    if not val_files:
        raise SystemExit("验证集为空: " + VAL_IMG)

    # B 速度
    print("=== 测试B: 推理速度 ===")
    speed = test_speed(imread(val_files[0]))
    json.dump(speed, open(os.path.join(OUT, "speed.json"), "w"))

    # A/C 精度 + 阈值
    print("=== 测试A/C: 验证集推理(%d张) ===" % len(val_files))
    dump_dir(VAL_IMG, os.path.join(OUT, "det_val.json"))
    val_res = eval_val(os.path.join(OUT, "det_val.json"), VAL_LBL)

    # D 背景误检
    print("=== 测试D: 背景误检 ===")
    dump_dir(BG_DIR, os.path.join(OUT, "det_bg.json"))
    bg_res = eval_bg(os.path.join(OUT, "det_bg.json"))

    # ---------- 汇总 results.md ----------
    md = []
    md.append("# 树莓派5 毛发检测模型验证结果\n")
    md.append("- 模型：hair_yolov8n_v3_20261016 FP16 NCNN（生产部署版）")
    md.append("- 推理参数：imgsz=1280，4线程，NMS IoU阈值0.45（与网页控制台一致）")
    md.append("- 验证集：108张（来自DMEOI，未参与训练）")
    md.append("- 背景测试：芝麻油全部子集 %d 张（无毛发、从未参与训练）\n" % bg_res["n_images"])

    md.append("## 测试B：推理速度（100帧计时）\n")
    md.append("| 指标 | 数值 |")
    md.append("|---|---|")
    md.append("| 平均单帧耗时 | **%.1f ms**（±%.1f ms） |" % (speed["mean_ms"], speed["std_ms"]))
    md.append("| 等效帧率 | **%.2f FPS** |" % speed["fps"])
    md.append("| 最快/最慢 | %.1f ms / %.1f ms |" % (speed["min_ms"], speed["max_ms"]))
    md.append("| 计时帧数 | %d 帧（前3帧预热不计） |\n" % speed["frames"])

    md.append("## 测试A：边缘端精度（验证集108张）\n")
    md.append("| 指标 | 树莓派5边缘端 | PC端参考 |")
    md.append("|---|---|---|")
    md.append("| 平均准确率 AP50 | **%.3f** | 0.993 |" % val_res["ap50"])
    md.append("\n（P/R 见下方阈值表；AP50为全点插值，与ultralytics口径可能略有差异）\n")

    md.append("## 测试C：置信度阈值扫描\n")
    md.append("| 阈值 | 查准率P | 查全率R | F1 | TP | FP | FN |")
    md.append("|---|---|---|---|---|---|---|")
    for thr, P, R, F1, TP, FP, FN in val_res["rows"]:
        mark = " ←F1最优" if thr == val_res["best_thr"] else ""
        md.append("| %.2f | %.3f | %.3f | %.3f | %d | %d | %d%s |" % (thr, P, R, F1, TP, FP, FN, mark))

    md.append("\n## 测试D：背景误检（%d张，无毛发）\n" % bg_res["n_images"])
    md.append("| 阈值 | 误报图片数 | 误报框数 | 误报率(图片) |")
    md.append("|---|---|---|---|")
    for thr, n_img, n_box in bg_res["rows"]:
        md.append("| %.2f | %d | %d | %.1f%% |" % (thr, n_img, n_box, 100.0*n_img/bg_res["n_images"]))

    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(md))
    json.dump({"speed": speed, "val": {"ap50": val_res["ap50"], "best_thr": val_res["best_thr"], "best_f1": val_res["best_f1"]}, "bg": bg_res},
              open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print("\n=== 完成，结果在 ~/yolo/eval/results.md ===")
