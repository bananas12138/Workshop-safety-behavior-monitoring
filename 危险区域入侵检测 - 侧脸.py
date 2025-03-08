import cv2

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

    # 加载 Haar Cascade 模型
    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    if frontal_face_cascade.empty() or profile_face_cascade.empty():
        print("加载 Haar Cascade 模型失败")
        return

    # 定义危险区域 (例如：假设摄像头分辨率为 640x480，此处定义画面中央区域为危险区域)
    danger_zone = (220, 140, 200, 200)  # (x, y, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        # 1. 检测正脸
        faces_frontal = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if faces_frontal is not None:
            faces.extend(faces_frontal)

        # 2. 检测侧脸
        faces_profile = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if faces_profile is not None:
            faces.extend(faces_profile)

        # 3. 对翻转图像进行侧脸检测（用于检测另一侧脸）
        gray_flipped = cv2.flip(gray, 1)
        faces_profile_flipped = profile_face_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if faces_profile_flipped is not None:
            for (x, y, w, h) in faces_profile_flipped:
                # 翻转坐标到原图：原图宽度 - x - 宽度
                x_orig = gray.shape[1] - x - w
                faces.append((x_orig, y, w, h))

        # 在画面上绘制危险区域（红色矩形）
        dx, dy, dw, dh = danger_zone
        cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), (0, 0, 255), 2)

        danger_detected = False

        # 绘制检测到的人脸并判断是否与危险区域重叠
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if check_intersection((x, y, w, h), danger_zone):
                danger_detected = True

        if danger_detected:
            print("危险警告：检测到人员进入危险区域")
            cv2.putText(frame, "WARNING: Danger Zone Breached!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Camera - Haar Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
