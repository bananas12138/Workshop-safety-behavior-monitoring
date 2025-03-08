import cv2
import torch

def main():
    # 加载自定义YOLOv5模型（请确认文件路径正确）
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='helmet_head_person_l.pt', force_reload=False)
    model.eval()
    model.conf = 0.5  # 设置置信度阈值，可根据需要调整

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 使用YOLOv5模型进行推理，传入BGR图像即可
        results = model(frame)
        # YOLOv5输出格式：results.xyxy[0]的每一行格式为 [x1, y1, x2, y2, confidence, class]
        detections = results.xyxy[0]

        # 遍历检测结果
        for *xyxy, conf, cls in detections:
            # 将坐标转换为整数
            x1, y1, x2, y2 = map(int, xyxy)
            label = model.names[int(cls)]
            # 如果检测到的对象标签为"helmet"，则认为佩戴了安全帽，框出绿色框
            if label.lower() == 'helmet':
                color = (0, 255, 0)  # 绿色
                text = f"Helmet {conf:.2f}"
            else:
                # 其他类别认为未佩戴安全帽，此处用红色框，并在终端输出警告
                color = (0, 0, 255)  # 红色
                text = f"{label} {conf:.2f}"
                print("警告：检测到未戴头盔的人脸！")

            # 绘制检测框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示摄像头画面
        cv2.imshow("Helmet & Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
