from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载预训练的轻量化 YOLOv8n 模型作为起点
    model = YOLO("yolov8n.pt")

    # 2. 开始训练毛发检测模型（数据集：大豆油+菜籽油 540 张 + 60 张背景负样本 = 600 张）
    # data 指向转换好的 data.yaml（train 438 / val 108 / test 54；负样本仅进 train）
    # epochs 为训练轮数，数据量涨到 4.4 倍，轮数从 100 提到 150（收敛后由早停结束）
    # imgsz 是输入图片尺寸，毛发又细又长，用 1280 保留细节（嫌慢可以降到 1024，约快 1.5 倍）
    # batch 是批大小：1280 分辨率下 6G 显存只能到 4。这是"分辨率换批大小"的取舍，
    #   要加大 batch 只能降 imgsz。好在 ultralytics 自动按 nbs=64 做梯度累积，
    #   CUDA 下也默认开混合精度，这两项已在缓解小 batch 的影响
    # patience 为早停：连续 50 轮没有提升就自动结束，省时间
    model.train(
        data="../dataset/DMEOI_hair_yolo/data.yaml",  # 相对 scripts/ 的仓库内路径
        epochs=150,
        imgsz=1280,
        batch=4,
        patience=50,
        device='0',
        seed=0,               # 固定随机种子，保证多次训练结果可比

        # 438 张训练图全部缓存进内存，省掉每轮反复读盘（约占 1.7 GB 内存）
        cache=True, workers=4,

        # 余弦退火：训练更长时末段收敛更细
        lr0=0.01, cos_lr=True, warmup_epochs=3,

        # 轻量正则，抑制过拟合
        label_smoothing=0.1, weight_decay=0.001,

        # 加大框回归损失权重：小目标漏检 → box 7.5→10；定位精度不足 → dfl 1.5→2.0
        box=10.0, dfl=2.0,

        # 数据增强（针对细长目标，这套已验证有效，保持不动）
        mosaic = 0.5, scale = 0.2, flipud = 0.5,
        hsv_h = 0.005, hsv_s = 0.3, hsv_v = 0.3,
        close_mosaic = 25,    # 最后 25 轮关闭 mosaic（默认 10 轮，早停时往往来不及触发）
    )
