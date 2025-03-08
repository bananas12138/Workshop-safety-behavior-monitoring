import cv2


def check_intersection(rect1, rect2):
    """
    检查两个矩形（格式均为 (x, y, w, h)）是否有交集。
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 计算两个矩形的右下角坐标
    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2

    # 计算交集区域的宽度和高度
    overlap_w = max(0, min(x1_br, x2_br) - max(x1, x2))
    overlap_h = max(0, min(y1_br, y2_br) - max(y1, y2))

    return overlap_w * overlap_h > 0


def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化 HOG 行人检测器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 定义危险区域 (这里以一个固定矩形为例，需根据实际摄像头分辨率调整)
    # 假设摄像头分辨率为 640x480，此处定义画面中央区域为危险区域
    danger_zone = (220, 140, 200, 200)  # (x, y, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 检测人体，detectMultiScale 返回检测到的人体矩形框列表
        rects, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        # 在画面上绘制危险区域（红色矩形）
        dx, dy, dw, dh = danger_zone
        cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)

        danger_detected = False

        # 遍历检测到的人体
        for (x, y, w, h) in rects:
            # 绘制人体检测框（绿色矩形）
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 检查该人体框与危险区域是否有交集
            if check_intersection((x, y, w, h), danger_zone):
                danger_detected = True

        # 如果检测到危险区域内有人体，则在终端打印警告并在画面上显示警告信息
        if danger_detected:
            print("危险警告：检测到人体进入危险区域")
            cv2.putText(frame, "WARNING: Danger Zone Breached!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示当前画面
        cv2.imshow("Camera", frame)

        # 按 'q' 键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
