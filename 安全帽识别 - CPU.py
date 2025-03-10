import cv2
import torch

def compute_iou(box1, box2):
    """
    计算两个检测框的IoU（交并比），输入格式为 (x1, y1, x2, y2)
    如果有任何交汇，IoU就大于0
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def main():
    # 加载自定义YOLOv5模型（请确认文件路径正确）
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='helmet_head_person_l.pt', force_reload=False)
    model.eval()
    model.conf = 0.5  # 设置置信度阈值

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

        # 使用YOLOv5模型进行推理（传入BGR图像即可）
        results = model(frame)
        # YOLOv5输出格式：results.xyxy[0]的每一行格式为 [x1, y1, x2, y2, confidence, class]
        detections = results.xyxy[0]

        helmet_detections = []   # 佩戴安全帽（绿框）
        nohelmet_detections = [] # 未戴安全帽（红框）

        # 遍历检测结果并分类
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            label = model.names[int(cls)]
            if label.lower() == 'helmet':
                helmet_detections.append((x1, y1, x2, y2, conf, label))
            else:
                nohelmet_detections.append((x1, y1, x2, y2, conf, label))

        # 绘制佩戴安全帽的检测结果（绿色框）
        for det in helmet_detections:
            x1, y1, x2, y2, conf, label = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 对未戴安全帽的检测结果进行重叠判断：
        # 如果红框与任一绿框有任何交汇（IoU > 0），则认为该目标实际上已佩戴安全帽，不显示红框及警告
        warning_printed = False  # 每帧仅打印一次警告
        for det in nohelmet_detections:
            x1, y1, x2, y2, conf, label = det
            overlap = False
            for h_det in helmet_detections:
                hx1, hy1, hx2, hy2, hconf, hlabel = h_det
                iou = compute_iou((x1, y1, x2, y2), (hx1, hy1, hx2, hy2))
                if iou > 0:  # 只要有交汇就认为重叠
                    overlap = True
                    break
            if not overlap:
                # 绘制红色检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if not warning_printed:
                    print("警告：检测到未戴头盔的人脸！")
                    warning_printed = True

        # 显示摄像头画面
        cv2.imshow("Helmet & Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
