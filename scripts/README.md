# 程序脚本说明 —— 面向食品/药品产线的异物识别与剔除模拟

## 一、脚本清单与功能

| 文件 | 功能 |
|---|---|
| convert_voc_to_yolo.py | 数据格式转换：将 DMEOI 的 VOC XML 标注解析为 YOLO 格式（适配多种目录布局，含过小框过滤） |
| train_hair.py | 模型训练：YOLOv8n 训练配置与调用（imgsz=1280、box=10/dfl=2、针对细长目标的增强） |
| predict_hair.py | 预测可视化：批量检测并绘制红色检测框、置信度与 Hair Count 计数 |
| test_false_positive.py | 误检专项测试：背景负样本误检统计 |
| infer_eval.py | 边缘端验证：精度（mAP50）/速度（100帧计时）/置信度阈值扫描/背景误检 四项测试 |
| web_console.py | 上位机（网页控制台）：Flask 服务，NCNN FP16 推理，三类检测源、参数即时调节、结果轮次化管理、剔除触发与灯控 |
| act-led.sh | 剔除指示助手脚本：控制树莓派5 板载 ACT 灯（低电平有效），部署于 /usr/local/bin/act-led |

## 二、上位机（网页控制台）运行环境

- **硬件**：树莓派 5（四核 Cortex-A76，4GB+）
- **系统**：Raspberry Pi OS 64-bit（桌面版）
- **Python 环境**：Python 3.11 虚拟环境（~/yolo-env），依赖：flask、ncnn 1.0.20240410+、opencv-python、numpy
- **模型**：FP16 NCNN 格式（model.ncnn.param / model.ncnn.bin），见仓库 `models/hair_yolov8n_v3_20261016_fp16_ncnn_model/`
- **启动方式**：
  1. 手动：`~/yolo-env/bin/python ~/yolo/web_console.py`，浏览器访问 `http://192.168.137.31:8000`
  2. 生产形态：systemd 服务 `hair-console` 开机自启（通电+插网线后 1~2 分钟自动可用）
- **剔除指示**：act-led.sh 需复制到树莓派 `/usr/local/bin/act-led` 并在 sudoers 中免密放行

## 三、备注

- web_console.py 为部署日（2026-11-01）上传至树莓派的副本，与生产环境一致性已验证；树莓派在线时可通过 `scp keshe21@192.168.137.31:~/yolo/web_console.py .` 获取最新版
- 训练/预测脚本运行于 PC 端（RTX 3060 + CUDA），依赖 ultralytics、opencv
