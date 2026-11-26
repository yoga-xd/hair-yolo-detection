from ultralytics import YOLO
import cv2
import os
import glob
import hashlib

# ==================== 配置区（按需修改） ====================

# 1. 模型权重（用存档那份，避免 runs 目录被清理后失效）
model_path = "models/hair_yolov8n_v3_20261016.pt"  # 相对仓库根目录；也可改为绝对路径

# 2. 背景图目录（必须是"不含毛发"的图，可以写多个）
#    默认指向仓库内 background/ 目录（60 张无毛发背景图）；如需扩充可再加目录
bg_folders = [
    "dataset/DMEOI_hair_yolo/background",  # 无毛发背景图（DMEOI 非毛发子集，可自行扩充）
]
# 3. 训练集目录：脚本会自动排除其中出现过的图片，避免"用训练数据测误报"
#    改成 None 或 "" 可关闭排除
train_dataset = "dataset/DMEOI_hair_yolo"

# 4. 要统计的置信度阈值（跑一遍即可算出所有阈值的结果）
conf_list = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

# 5. 推理参数
imgsz = 1280
device = '0'

# 6. 输出目录：保存"有误报"的图，便于人工复核
#    注意：这个目录名带中文，所以写图必须用 imwrite_u（见下），不能用 cv2.imwrite
output_dir = "results/fp_test"  # 误检测试输出目录

# ==================== 以下不用改 ====================


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def imread_u(path):
    """读图（支持中文路径）"""
    import numpy as np
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    """写图（支持中文路径）。
    注意：cv2.imwrite 在中文路径下会静默失败（返回 False，不报错），
    必须用 imencode + tofile 绕过。"""
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def draw_red(image, boxes, confs):
    """红框 + 红色置信度数字"""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.6, min(w, h) / 1400.0)
    th = max(2, int(round(min(w, h) / 600.0)))
    for xyxy, c in zip(boxes, confs):
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), th)
        text = "%.2f" % float(c)
        ty = y1 - 10 if y1 - 10 > 20 else y2 + 30
        cv2.putText(image, text, (x1, ty), font, fs, (0, 0, 255), th, cv2.LINE_AA)
    return image


os.makedirs(output_dir, exist_ok=True)

# ---- 1. 建立"训练用过的图"指纹库，用于排除 ----
used = set()
if train_dataset and os.path.isdir(train_dataset):
    for f in glob.glob(os.path.join(train_dataset, 'images', '*', '*.jpg')):
        used.add(md5(f))
print("训练集指纹库：%d 张（这些图会被排除，避免自欺欺人）" % len(used))

print("=" * 70)
print("误检测试（在纯背景图上统计误报）")
print("=" * 70)
print("模型      : %s" % model_path)
print("推理尺寸  : %d    设备: %s" % (imgsz, device))
print("背景图目录: %d 个" % len(bg_folders))
print("-" * 70)

# ---- 2. 逐个目录推理 ----
# 重要：ultralytics 不能接收 Python list 作为 source —— 它会把 list 当成
# "一批已加载到内存的图像"，一次性堆叠送进网络（上千张必然爆显存）。
# 也不能传 .txt 清单 —— LoadImages 用 rsplit() 按空白切分，路径里的空格会被切断。
# 所以这里直接传"目录路径字符串"，一次一个目录，逐张读取，最稳妥。
model = YOLO(model_path)
per_image = []          # [(path, boxes, confs)]
n_seen = 0
n_skipped = 0

for folder in bg_folders:
    if not os.path.isdir(folder):
        print("[警告] 目录不存在，已跳过：%s" % folder)
        continue
    print("  处理中：%s" % os.path.basename(os.path.dirname(folder)))
    for r in model.predict(source=folder, conf=0.01, imgsz=imgsz,
                           device=device, verbose=False, stream=True):
        n_seen += 1
        if used and md5(r.path) in used:      # 排除训练用过的图
            n_skipped += 1
            continue
        if r.boxes is None or len(r.boxes) == 0:
            per_image.append((r.path, [], []))
        else:
            per_image.append((r.path,
                              r.boxes.xyxy.cpu().numpy().tolist(),
                              r.boxes.conf.cpu().numpy().tolist()))

n_img = len(per_image)
print("-" * 70)
print("共推理 %d 张，排除训练用图 %d 张，实际参与统计 %d 张" % (n_seen, n_skipped, n_img))
if n_img == 0:
    print("没有可统计的图，脚本退出。请检查 bg_folders。")
    raise SystemExit(0)

# ---- 3. 按不同阈值统计 ----
print("-" * 70)
print("%-10s %-16s %-14s %-14s" % ("置信度阈值", "有误报的图", "误报框总数", "平均每张误报"))
print("-" * 70)
for conf in conf_list:
    n_img_fp = 0
    n_box_fp = 0
    for _, _, confs in per_image:
        n = sum(1 for c in confs if c >= conf)
        if n > 0:
            n_img_fp += 1
            n_box_fp += n
    print("%-10.2f %-16s %-14d %-14.3f"
          % (conf,
             "%d (%.1f%%)" % (n_img_fp, 100.0 * n_img_fp / n_img),
             n_box_fp,
             n_box_fp / float(n_img)))

# ---- 4. 关键指标：最高误报置信度 ----
all_conf = [c for _, _, confs in per_image for c in confs]
print("-" * 70)
if all_conf:
    print("共检出 %d 个框，其中最高置信度 = %.3f" % (len(all_conf), max(all_conf)))
    print(">> 只要置信度阈值高于 %.2f，本测试集上就不会有任何误报。" % max(all_conf))
else:
    print(">> 全部背景图均未检出任何目标 —— 零误报")

# ---- 5. 保存误报图 + 误报清单 ----
lines = []
saved = 0
for path, boxes, confs in per_image:
    if not confs:
        continue
    show = [i for i, c in enumerate(confs) if c >= conf_list[0]]
    if not show:
        continue
    lines.append("%-14s 检出 %d 个框，最高置信度 %.3f  <- %s"
                 % (os.path.basename(path), len(show),
                    max(confs[i] for i in show), path))
    if saved < 40:
        img = imread_u(path)
        if img is not None:
            draw_red(img, [boxes[i] for i in show], [confs[i] for i in show])
            if imwrite_u(os.path.join(output_dir, os.path.basename(path)), img):
                saved += 1

# 清单用 Python 原生写文件，中文路径没问题
list_path = os.path.join(output_dir, "误报清单.txt")
with open(list_path, 'w', encoding='utf-8') as f:
    f.write("误检测试 · 误报清单（阈值 %.2f 以下）\n" % conf_list[0])
    f.write("共 %d 张图检出疑似误报，明细如下：\n" % len(lines))
    f.write("=" * 70 + "\n")
    f.write("\n".join(lines) + "\n")

print("-" * 70)
print("含误报的图共 %d 张，已保存标注图 %d 张" % (len(lines), saved))
print("图片目录：%s" % output_dir)
print("清单文件：%s" % list_path)
