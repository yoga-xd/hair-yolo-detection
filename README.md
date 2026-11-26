# 基于YOLOv8n的毛发异物检测与剔除模拟

面向食品/药品产线场景的机器视觉检测项目：以公开数据集构建毛发检测数据集，训练 YOLOv8n 单类目标检测模型，量化部署至树莓派 5 边缘推理，并通过 GPIO 触发板载指示灯模拟产线剔除动作，配套网页控制台上位机，形成"图像采集—异物识别—自动剔除—上位机监控"完整闭环。

## 核心指标

| 指标 | 数值 |
|------|------|
| 数据集规模 | 600 张（540 正样本 + 60 负样本，负样本占比 10%，7:2:1 划分） |
| 验证集查准率 / 召回率 | 0.998 / 0.982 |
| 平均准确率 mAP50（PC 端） | 0.993 |
| mAP50（树莓派 5，FP16 NCNN） | 0.991（与 PC 端一致，量化零精度损失） |
| 边缘端单帧耗时 | 324 ms（约 3.09 帧/秒，4 线程） |
| 部署置信度阈值 | 0.55～0.60（F1=0.987） |
| 背景误检（981 张无毛发图像） | 0 |

训练曲线与检测效果见 [`results/`](results/)。

## 系统组成

采集（手机/摄像头拍摄容器样本）→ 数据集构建与格式转换 → YOLOv8n 训练（RTX 3060）→ FP16 NCNN 量化导出 → 树莓派 5 边缘推理（letterbox + NMS，与训练端预处理一致）→ 检出触发 GPIO 点亮板载 ACT 指示灯（低电平有效，保持 1 秒，上升沿计数）→ 网页控制台上位机（Flask）。

## 仓库结构

```
hair-yolo-detection/
├── scripts/            # 全部程序脚本（见 scripts/README.md 详细说明）
│   ├── convert_voc_to_yolo.py   # VOC XML → YOLO txt 格式转换
│   ├── train_hair.py            # 训练配置与调用（imgsz=1280, box=10, dfl=2）
│   ├── predict_hair.py          # 批量检测可视化（红框 + 置信度 + 计数）
│   ├── test_false_positive.py   # 背景负样本误检专项测试
│   ├── infer_eval.py            # 边缘端精度/速度/阈值扫描/误检 四项验证
│   ├── web_console.py           # 上位机：Flask 网页控制台（NCNN FP16 推理）
│   └── act-led.sh               # 树莓派板载 ACT 灯控制脚本
├── models/             # 训练权重（.pt）与 FP16 NCNN 部署模型
├── results/            # 训练曲线、PR 曲线、混淆矩阵、检测效果图
├── dataset/            # YOLO 格式数据集（images/labels + data.yaml）
├── docs/               # 模型训练报告
└── demo/               # 演示视频（采集素材与检测输出）
```

## 环境与复现

**训练（PC 端）**

```bash
pip install ultralytics
python scripts/train_hair.py          # 数据路径在脚本内配置区修改
python scripts/predict_hair.py        # 批量检测可视化
python scripts/test_false_positive.py # 背景误检测试
```

**边缘部署（树莓派 5，Raspberry Pi OS 64-bit）**

```bash
python -m venv ~/yolo-env && source ~/yolo-env/bin/activate
pip install flask ncnn opencv-python numpy
# 拷贝 models/ 与 scripts/web_console.py 到树莓派 ~/yolo/
python ~/yolo/web_console.py          # 浏览器访问 http://<树莓派IP>:8000
# 剔除指示：cp scripts/act-led.sh /usr/local/bin/act-led 并 sudo chmod +x
```

## 数据集说明

- 图像与标注来自公开数据集 **DMEOI**（Dual-modal Edible Oil Impurity Dataset）的毛发（hair）子集，在此基础上加入背景负样本并重新划分。
- `data.yaml` 不含绝对路径，训练前无需修改；`dataset/DMEOI_hair_yolo/background/` 内为 60 张无毛发背景图，供误检测试使用。
- 标注为 YOLO 单类格式：`类别号 中心点x 中心点y 宽 高`（均为归一化坐标）。
- 划分与格式转换可用 `scripts/convert_voc_to_yolo.py` 从 VOC 原始标注复现。

**原始数据集引用**（许可 CC BY 4.0，使用或再分发须注明出处，以下格式即满足要求）

论文：

> Wang H, Chen Q, Zhao J, et al. Dual-modal edible oil impurity dataset for weak feature detection[J]. Scientific Data, 2024, 11: 1426. DOI: [10.1038/s41597-024-04305-w](https://doi.org/10.1038/s41597-024-04305-w)

数据集（figshare 官方引用格式，v1，2025-04-11 发布）：

> Wang, Huiyu (2025). *DMEOI: Dual-modal edible oil impurity dataset*. figshare. Dataset. DOI: [10.6084/m9.figshare.25721409.v1](https://doi.org/10.6084/m9.figshare.25721409.v1)

数据集主页：[figshare.com/articles/dataset/DMEOI_Dual-modal_edible_oil_impurity_dataset/25721409](https://figshare.com/articles/dataset/DMEOI_Dual-modal_edible_oil_impurity_dataset/25721409)

## 演示视频

`demo/` 内含两段视频：采集素材（容器内壁贴附毛发的实拍）与检测输出（红框置信度 + 计数）。

## License

本项目代码以 [MIT License](LICENSE) 发布。数据集部分引用自 DMEOI（[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)），再分发或修改须按其要求注明原始出处（见上文引用格式）。
