import cv2
import numpy as np

def check_intersection(rect1, rect2):
    """
    检查两个矩形（格式均为 (x, y, w, h)）是否有交集。
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2
    overlap_w = max(0, min(x1_br, x2_br) - max(x1, x2))
    overlap_h = max(0, min(y1_br, y2_br) - max(y1, y2))
    return overlap_w * overlap_h > 0

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化人体检测器：HOG + SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 初始化 Haar Cascade 人脸检测器（正脸和侧脸）
    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    if frontal_face_cascade.empty() or profile_face_cascade.empty():
        print("加载 Haar Cascade 模型失败")
        return

    # 定义危险区域（这里以画面中央的一块矩形区域为例，注意根据实际分辨率调整）
    danger_zone = (220, 140, 200, 200)  # (x, y, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 绘制危险区域（红色矩形）
        dx, dy, dw, dh = danger_zone
        cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)

        danger_detected = False

        # 1. 人体检测（HOG检测器）
        human_rects, _ = hog.detectMultiScale(frame, winStride=(8, 8))
        for (x, y, w, h) in human_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if check_intersection((x, y, w, h), danger_zone):
                danger_detected = True

        # 转换为灰度图用于 Haar 检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 正脸检测
        faces_frontal = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces_frontal:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if check_intersection((x, y, w, h), danger_zone):
                danger_detected = True

        # 3. 侧脸检测（原图检测）
        faces_profile = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces_profile:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            if check_intersection((x, y, w, h), danger_zone):
                danger_detected = True

        # 4. 侧脸检测（翻转图像检测另一侧）
        gray_flipped = cv2.flip(gray, 1)
        faces_profile_flipped = profile_face_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces_profile_flipped:
            # 将翻转后的坐标转换回原图坐标
            x_orig = gray.shape[1] - x - w
            cv2.rectangle(frame, (x_orig, y), (x_orig + w, y + h), (255, 255, 0), 2)
            if check_intersection((x_orig, y, w, h), danger_zone):
                danger_detected = True

        # 若任何检测到的目标与危险区域重叠，输出警告
        if danger_detected:
            print("危险警告：检测到目标进入危险区域")
            cv2.putText(frame, "WARNING: Danger Zone Breached!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Integrated Detection", frame)

        # 按 'q' 键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
