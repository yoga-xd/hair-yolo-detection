#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOC -> YOLO 格式转换脚本（零依赖，仅用 Python 标准库）

专为 DMEOI 数据集设计，但也可用于任意 VOC 风格数据集。

特点：
  1. 自动适配多种目录布局：
     - <class>/JEPGImages  + <class>/Annotations   （注意原数据集把 JPEGImages 拼成了 JEPGImages）
     - <class>/JPEGImages  + <class>/Annotations
     - VOC2007/{JPEGImages, Annotations}
     - 图片与 XML 混放在同一目录（event/denoised 那种）
  2. 可只抽取指定类别（默认只取 hair），自动忽略其他类别。
  3. 归一化坐标自动换算并裁剪到 [0,1]。
  4. 可选把"无目标类别"的图作为背景负样本一起输出。
  5. 自动划分 train/val/test，并可做内容去重（避免 event 与 full-pic 的重复图）。
  6. 会读取图片真实尺寸并与 XML 中声明的尺寸比对，不一致时给出告警。

用法示例：
  # 先看一眼统计，不写文件
  python convert_voc_to_yolo.py --dry-run

  # 正式转换（只取 hair 类，输出到 D:\\DMEOI_hair_yolo）
  python convert_voc_to_yolo.py --src <VOC数据集目录> --dst DMEOI_hair_yolo

  # 同时把无毛发图当负样本，并只处理 full-pic 模态
  python convert_voc_to_yolo.py --include-background --modes full-pic
"""

import argparse
import hashlib
import os
import random
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------------------------------------------- 图片尺寸读取

def _jpeg_size(f):
    """从 JPEG 文件头解析尺寸，返回 (w, h)，失败返回 None。"""
    f.seek(0)
    data = f.read()
    i = 2
    n = len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0~SOF15（排除 DHT=0xC4, JPG=0xC8, DAC=0xCC）
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = struct.unpack(">H", data[i + 5:i + 7])[0]
            w = struct.unpack(">H", data[i + 7:i + 9])[0]
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None


def _png_size(f):
    """从 PNG 文件头解析尺寸。"""
    f.seek(16)
    data = f.read(8)
    if len(data) < 8:
        return None
    w, h = struct.unpack(">II", data)
    return w, h


def get_image_size(path):
    """读取图片真实宽高，失败返回 None。"""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            f.seek(0)
            if head[:3] == b"\xff\xd8\xff":
                return _jpeg_size(f)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return _png_size(f)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 路径工具

def find_image_for_xml(xml_path):
    """
    给定一个 XML 路径，返回对应的图片路径（找不到返回 None）。
    支持：同目录同名、Annotations <-> JPEGImages/JEPGImages 兄弟目录。
    """
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    xml_dir = os.path.dirname(xml_path)
    parent = os.path.dirname(xml_dir)
    cur_name = os.path.basename(xml_dir)

    candidate_dirs = [xml_dir]
    if cur_name.lower() == "annotations":
        # 兄弟目录里找常见图片目录名
        for sib in ("JPEGImages", "JEPGImages", "images", "Images", "JPEGimages"):
            candidate_dirs.append(os.path.join(parent, sib))
        # 兜底：父目录下所有目录
        try:
            for d in os.listdir(parent):
                p = os.path.join(parent, d)
                if os.path.isdir(p):
                    candidate_dirs.append(p)
        except OSError:
            pass

    for d in candidate_dirs:
        for ext in IMG_EXTS:
            for name in (stem + ext, stem + ext.upper()):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    return p
    return None


def is_under(path, mode_names):
    """判断路径是否位于指定模态目录（如 event / full-pic）下。"""
    if not mode_names:
        return True
    parts = os.path.normpath(path).replace("\\", "/").lower().split("/")
    return any(m.lower() in parts for m in mode_names)


# ---------------------------------------------------------------- 核心解析

def parse_voc(xml_path, target_classes):
    """
    解析 VOC XML。
    返回 (size_from_xml, objects)，objects 为 [(cls_name, xmin, ymin, xmax, ymax), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    size_from_xml = None
    if size_el is not None:
        try:
            size_from_xml = (int(float(size_el.findtext("width"))),
                             int(float(size_el.findtext("height"))))
        except (TypeError, ValueError):
            size_from_xml = None

    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if target_classes and name not in target_classes:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        objects.append((name, xmin, ymin, xmax, ymax))
    return size_from_xml, objects


def to_yolo_lines(objects, w, h, class_to_id, min_size=1.0):
    """把 VOC 框转成 YOLO 归一化行；过滤掉过小的框。"""
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        # 裁剪到图像范围内
        xmin = max(0.0, min(xmin, w))
        xmax = max(0.0, min(xmax, w))
        ymin = max(0.0, min(ymin, h))
        ymax = max(0.0, min(ymax, h))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw < min_size or bh < min_size:
            continue

        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        nw = bw / w
        nh = bh / h
        # 再夹一次，防止浮点误差越界
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        lines.append(f"{class_to_id[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(
        description="VOC -> YOLO 转换（DMEOI 数据集适配版）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="", help="VOC 数据集根目录（必填）")
    ap.add_argument("--dst", default="DMEOI_hair_yolo", help="输出目录（相对当前目录）")
    ap.add_argument("--classes", default="hair",
                    help="要抽取的类别，逗号分隔（默认 hair）")
    ap.add_argument("--include-background", action="store_true",
                    help="把不含目标类别的图作为背景负样本一起输出（生成空标签文件）")
    ap.add_argument("--bg-max", type=int, default=0,
                    help="背景负样本最多取多少张（0=不限；仅配合 --include-background 生效）")
    ap.add_argument("--bg-train-only", action="store_true",
                    help="背景负样本只加进 train 划分（val/test 保持纯正样本，便于实验对比）")
    ap.add_argument("--bg-exclude", default="VOC2007",
                    help="路径含该子串的图不作背景负样本（默认排除只标粗类 impurity 的 VOC2007 目录）")
    ap.add_argument("--modes", default="full-pic",
                    help="只处理这些模态目录，逗号分隔；留空表示全部（默认 full-pic）")
    ap.add_argument("--split", default="0.7,0.2,0.1",
                    help="train,val,test 比例（默认 0.7,0.2,0.1）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--mode-op", default="copy", choices=["copy", "move", "none"],
                    help="对图片的操作：copy/move/none（默认 copy）")
    ap.add_argument("--dedupe", action="store_true", default=True,
                    help="按文件内容去重（默认开启）")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                    help="关闭内容去重")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")

    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_to_id = {c: i for i, c in enumerate(target_classes)}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bg_exclude_list = [p.strip() for p in args.bg_exclude.split(",") if p.strip()]

    if not os.path.isdir(src):
        print(f"[错误] 源目录不存在：{src}")
        return 1

    try:
        ratios = [float(x) for x in args.split.split(",")]
        assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6
    except Exception:
        print("[错误] --split 需为三个和为 1 的数，例如 0.7,0.2,0.1")
        return 1

    print("=" * 68)
    print("VOC -> YOLO 转换")
    print("=" * 68)
    print(f"源目录   : {src}")
    print(f"输出目录 : {dst}{'  (dry-run，不写文件)' if args.dry_run else ''}")
    print(f"目标类别 : {target_classes}")
    print(f"处理模态 : {modes if modes else '全部'}")
    print(f"划分比例 : train={ratios[0]}  val={ratios[1]}  test={ratios[2]}")
    print("-" * 68)

    # 1) 收集所有 XML
    xml_files = []
    for root_dir, _dirs, files in os.walk(src):
        if not is_under(root_dir, modes):
            continue
        for fn in files:
            if fn.lower().endswith(".xml"):
                xml_files.append(os.path.join(root_dir, fn))
    print(f"发现 XML 文件：{len(xml_files)} 个")

    # 2) 解析
    samples = []                      # (img_path, lines, counters)
    bg_pool = []                      # 候选背景负样本（稍后按 --bg-max 采样）
    bg_excluded = 0                   # 因落在 --bg-exclude 名单里而被跳过的背景
    cls_counter = Counter()           # 各类别框数量
    skipped_no_img = 0
    skipped_no_target = 0
    size_mismatch = 0
    missing_size = 0
    seen_hashes = {}
    dup = 0

    for xp in xml_files:
        img = find_image_for_xml(xp)
        if img is None:
            skipped_no_img += 1
            continue

        try:
            size_xml, objs_all = parse_voc(xp, None)      # 先全取，用于统计
        except ET.ParseError:
            skipped_no_img += 1
            continue

        for _n, *_ in objs_all:
            cls_counter[_n] += 1

        # 只保留目标类
        objs = [o for o in objs_all if o[0] in class_to_id]
        is_bg = not objs
        if is_bg:
            if not args.include_background:
                skipped_no_target += 1
                continue
            # 粗标/来源不可靠的目录直接排除，避免把实际含毛发的图当成负样本
            if bg_exclude_list and any(p.lower() in xp.lower() for p in bg_exclude_list):
                bg_excluded += 1
                continue

        # 尺寸：优先真实尺寸，失败则用 XML 声明
        real = get_image_size(img)
        if real is None:
            missing_size += 1
            size = size_xml
        else:
            size = real
            if size_xml and real != size_xml:
                size_mismatch += 1
        if not size or size[0] <= 0 or size[1] <= 0:
            skipped_no_target += 1
            continue
        w, h = size

        # 去重
        if args.dedupe:
            try:
                with open(img, "rb") as f:
                    digest = hashlib.md5(f.read()).hexdigest()
            except OSError:
                digest = None
            if digest:
                if digest in seen_hashes:
                    dup += 1
                    continue
                seen_hashes[digest] = img

        lines = to_yolo_lines(objs, w, h, class_to_id)
        (bg_pool if is_bg else samples).append({"img": img, "lines": lines})

    if args.include_background and bg_pool:
        random.seed(args.seed)
        if args.bg_max and len(bg_pool) > args.bg_max:
            bg_pool = random.sample(bg_pool, args.bg_max)
        if not args.bg_train_only:
            samples.extend(bg_pool)

    n_pos = sum(1 for s in samples if s["lines"])
    n_bg = len(bg_pool)
    print(f"有效样本：{len(samples) + (n_bg if args.bg_train_only else 0)}"
          f"（含目标类别 {n_pos} ｜ 背景负样本 {n_bg}"
          + ("，仅加入 train）" if (args.include_background and args.bg_train_only) else "）"))
    print(f"跳过：无对应图片 {skipped_no_img} ｜ 无目标类别 {skipped_no_target}"
          + (f" ｜ 背景排除 {bg_excluded}" if args.include_background else "")
          + (f" ｜ 内容重复 {dup}" if args.dedupe else ""))
    if size_mismatch:
        print(f"[告警] 有 {size_mismatch} 张图片的真实尺寸与 XML 声明不一致，已按真实尺寸换算")
    if missing_size:
        print(f"[告警] 有 {missing_size} 张无法读取真实尺寸，已回退使用 XML 声明")

    print("-" * 68)
    print("全库标注类别统计（所有 XML 中的 object 数量）：")
    for name, cnt in cls_counter.most_common():
        mark = "  <-- 目标类别" if name in class_to_id else ""
        print(f"  {name:<16} {cnt}{mark}")

    if not samples:
        print("\n没有可转换的样本，已退出。")
        return 0

    if args.dry_run:
        print("-" * 68)
        print("dry-run 结束，未写入任何文件。去掉 --dry-run 即可正式转换。")
        return 0

    # 3) 划分
    random.seed(args.seed)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    # 背景负样本只进训练集时，在此并入（val / test 保持纯正样本）
    if args.bg_train_only and bg_pool:
        splits["train"].extend(bg_pool)

    # 4) 写出
    for sub in ("train", "val", "test"):
        os.makedirs(os.path.join(dst, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(dst, "labels", sub), exist_ok=True)

    written = Counter()
    for sub, items in splits.items():
        for i, s in enumerate(items):
            src_img = s["img"]
            ext = os.path.splitext(src_img)[1].lower()
            base = os.path.splitext(os.path.basename(src_img))[0]
            # 加序号前缀，避免不同子目录同名文件互相覆盖
            new_name = f"{i:05d}_{base}"
            dst_img = os.path.join(dst, "images", sub, new_name + ext)
            dst_lbl = os.path.join(dst, "labels", sub, new_name + ".txt")

            if args.mode_op == "copy":
                shutil.copy2(src_img, dst_img)
            elif args.mode_op == "move":
                shutil.move(src_img, dst_img)

            with open(dst_lbl, "w", encoding="utf-8") as f:
                if s["lines"]:
                    f.write("\n".join(s["lines"]) + "\n")
            written[sub] += 1

    # 5) 写 data.yaml
    yaml_path = os.path.join(dst, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dst}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(target_classes)}\n")
        f.write("names:\n")
        for c in target_classes:
            f.write(f"  {class_to_id[c]}: {c}\n")

    print("-" * 68)
    print("转换完成：")
    for sub in ("train", "val", "test"):
        print(f"  {sub:<6} {written[sub]:>4} 张")
    print(f"  data.yaml -> {yaml_path}")
    print(f"\n可直接用于训练：\n  yolo detect train model=yolov8n.pt data=\"{yaml_path}\" epochs=200 imgsz=640")
    return 0


if __name__ == "__main__":
    sys.exit(main())
