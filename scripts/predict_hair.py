from ultralytics import YOLO
import cv2
import os

# 1. 加载训练好的毛发检测模型
# 训练脚本 train_hair.py 跑完后，最佳权重默认在 runs/detect/train/weights/best.pt
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 设置文件夹路径
# 待检测图片所在目录（换成你自己的图片文件夹即可）
folder_path = "dataset/DMEOI_hair_yolo/images/val"  # 仓库内验证集示例
# 设置自定义结果的保存路径
output_dir = "runs/detect/predict_hair"
os.makedirs(output_dir, exist_ok=True)

# 提前读取文件夹内所有的图片文件名（排序后与推理结果的顺序保持一致）
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.tif', '.TIF')
img_names = sorted(f for f in os.listdir(folder_path) if f.endswith(valid_extensions))

# 推理时关闭原生自动保存
results = model.predict(
    source=folder_path,
    save=False,
    conf=0.5
)

print("\n========== 开始批量检测 ==========")

# 3. 遍历整组图片的检测结果 (通过 zip 将结果和文件名绑定)
for img_name, result in zip(img_names, results):

    img_path = os.path.join(folder_path, img_name)
    annotated_frame = cv2.imread(img_path)

    if annotated_frame is None:
        print(f"[ERROR] 无法读取图片 {img_name}，已跳过")
        continue

    boxes = result.boxes

    # 初始化当前图片中毛发的计数器
    hair_count = 0

    if len(boxes) == 0 or boxes is None:
        print(f"[INFO] 图片 [{img_name}]: 未检测到毛发")
    else:
        for box in boxes:
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            hair_count += 1

            # 红色矩形框：把毛发框出来
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)

            # 红色字体：在框上方标出该框的置信度（框贴顶时改画在框内下方，避免文字被裁掉）
            ty = y1 - 10 if y1 - 10 > 20 else y2 + 30
            cv2.putText(annotated_frame, f"hair {conf:.2f}", (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)

            print(f"[INFO] 图片 [{img_name}] -> 检出毛发, 置信度: {conf:.2f}")

    # 🚀 在图片的左上角绘制毛发检出个数（有毛发标红，无毛发标绿）
    count_color = (0, 0, 255) if hair_count > 0 else (0, 255, 0)
    cv2.putText(
        img=annotated_frame, text=f"Hair Count: {hair_count}", org=(40, 100),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=count_color, thickness=6, lineType=cv2.LINE_AA
    )

    # 保存标注后的图片
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, annotated_frame)

print(f"\n========== 批量检测完成 ==========")
print(f"结果保存文件夹: {output_dir}")
