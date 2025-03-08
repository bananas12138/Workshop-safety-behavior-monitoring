# 导入 OpenCV 库，用于图像处理和视频捕捉
import cv2
# 导入 NumPy 库，用于处理数组和多边形数据
import numpy as np
# 导入 logging 模块，用于记录日志信息
import logging
# 导入 time 模块（此示例中未直接使用，可用于后续扩展，如时间戳处理）
import time

# 配置日志记录，设置日志级别为 DEBUG，并定义日志格式和时间格式
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG，记录所有级别的日志
    format='[%(asctime)s] [%(levelname)s] %(message)s',  # 日志格式：时间戳、日志级别和日志消息
    datefmt='%H:%M:%S'  # 时间格式为 时:分:秒
)


# 定义人脸检测类，用于封装人脸检测的相关操作
class FaceDetector:
    # 构造函数，初始化人脸检测器
    def __init__(self, cascade_path=None):
        # 如果未指定级联模型路径，则使用 OpenCV 内置的人脸检测模型路径
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        # 加载 Haar 级联模型，用于人脸检测
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        # 如果加载失败，则抛出异常并提示错误信息
        if self.face_cascade.empty():
            raise IOError("无法加载人脸检测模型：{}".format(cascade_path))
        # 记录模型加载成功的信息
        logging.info("FaceDetector 初始化完成，使用模型：{}".format(cascade_path))

    # 定义检测人脸的方法，接收一帧图像以及检测参数
    def detect_faces(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        # 将彩色图像转换为灰度图像，提高检测效率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用 Haar 级联检测人脸，返回人脸区域的矩形坐标列表
        faces = self.face_cascade.detectMultiScale(
            gray,  # 灰度图像
            scaleFactor=scaleFactor,  # 每次图像尺寸缩放比例
            minNeighbors=minNeighbors,  # 每个候选矩形保留多少个邻近矩形才算真的人脸
            minSize=minSize  # 最小的人脸尺寸
        )
        # 记录检测到的人脸数量
        logging.debug("检测到 {} 个人脸".format(len(faces)))
        # 返回检测到的人脸列表，每个元素为 (x, y, w, h)
        return faces


# 定义危险区域类，用于封装危险区域的相关操作
class DangerZone:
    # 构造函数，初始化危险区域，多边形的顶点由 polygon_points 指定
    def __init__(self, polygon_points):
        """
        polygon_points: [(x1, y1), (x2, y2), ...] 定义危险区域的顶点坐标
        """
        # 将顶点列表转换为 NumPy 数组，数据类型为整型
        self.polygon = np.array(polygon_points, dtype=np.int32)
        # 记录危险区域初始化成功的信息
        logging.info("DangerZone 初始化完成，区域坐标：{}".format(polygon_points))

    # 定义方法，在图像上绘制危险区域边界
    def draw_zone(self, frame, color=(0, 0, 255), thickness=2):
        # 使用 cv2.polylines 绘制多边形边界，参数 isClosed=True 表示闭合多边形
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=thickness)

    # 定义方法，判断某个点是否在危险区域内
    def is_inside(self, point):
        """
        判断一个点是否位于危险区域内
        point: (x, y)
        返回：True 如果点在区域内部或边界上，否则 False
        """
        # 将点转换为浮点型的元组，避免 cv2.pointPolygonTest 解析数据类型错误
        pt = (float(point[0]), float(point[1]))
        # 使用 cv2.pointPolygonTest 测试点的位置：返回正值表示在内部，0 表示在边界上，负值表示在外部
        result = cv2.pointPolygonTest(self.polygon, pt, False)
        # 如果结果大于等于 0，则表示点在区域内或在边界上，返回 True；否则返回 False
        return result >= 0


# 定义视频处理类，用于整合视频捕捉、人脸检测以及危险区域判断的功能
class VideoProcessor:
    # 构造函数，初始化视频处理器
    def __init__(self, face_detector, danger_zone=None, video_source=0):
        # 保存传入的人脸检测器对象
        self.face_detector = face_detector
        # 保存传入的危险区域对象（可以为 None）
        self.danger_zone = danger_zone
        # 打开指定的视频源，默认为摄像头（设备编号 0）
        self.cap = cv2.VideoCapture(video_source)
        # 如果视频源无法打开，则抛出异常
        if not self.cap.isOpened():
            raise IOError("无法打开视频源：{}".format(video_source))
        # 设置一个标志，控制视频处理的主循环是否继续运行
        self.running = True
        # 记录视频处理器初始化成功的信息
        logging.info("VideoProcessor 初始化完成，视频源：{}".format(video_source))

    # 定义主处理方法，实时捕捉视频帧并处理
    def process(self):
        """主循环：读取视频帧、检测人脸、判断是否进入危险区域，并实时显示结果"""
        while self.running:
            # 读取视频流中的一帧
            ret, frame = self.cap.read()
            # 如果未成功读取到帧，记录错误并退出循环
            if not ret:
                logging.error("无法读取视频帧")
                break

            # 使用人脸检测器检测当前帧中的人脸
            faces = self.face_detector.detect_faces(frame)

            # 如果定义了危险区域，则在图像上绘制危险区域的边界（蓝色线条）
            if self.danger_zone is not None:
                self.danger_zone.draw_zone(frame, color=(255, 0, 0), thickness=2)

            # 遍历检测到的每个脸部区域
            for (x, y, w, h) in faces:
                # 计算人脸的中心点坐标
                face_center = (x + w // 2, y + h // 2)
                # 判断人脸中心是否在危险区域内
                if self.danger_zone is not None and self.danger_zone.is_inside(face_center):
                    box_color = (0, 0, 255)  # 红色表示人脸在危险区域内
                    # 记录警告信息，提示检测到危险区域内的人脸
                    logging.warning("检测到危险区域内的人员"
                                    "，中心坐标：{}".format(face_center))
                else:
                    box_color = (0, 255, 0)  # 绿色表示人脸在安全区域内
                # 在人脸区域绘制矩形框，颜色根据是否处于危险区域决定
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                # 绘制人脸中心点的小圆点，便于观察
                cv2.circle(frame, face_center, 3, box_color, -1)

            # 在窗口中显示处理后的图像
            cv2.imshow("高级人脸检测", frame)
            # 检测是否按下 'q' 键，如按下则退出主循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False  # 更新运行标志，退出循环

        # 循环结束后，释放视频捕捉设备
        self.cap.release()
        # 关闭所有 OpenCV 创建的窗口
        cv2.destroyAllWindows()


# 定义主函数，用于初始化各个组件并启动视频处理
def main():
    # 定义危险区域的多边形顶点（根据实际场景可进行调整）
    danger_zone_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    # 初始化危险区域对象
    danger_zone = DangerZone(danger_zone_points)

    # 初始化人脸检测器对象
    face_detector = FaceDetector()

    # 初始化视频处理器对象，传入人脸检测器和危险区域对象
    video_processor = VideoProcessor(face_detector, danger_zone)

    try:
        # 启动视频处理的主循环
        video_processor.process()
    except Exception as e:
        # 如果在视频处理过程中出现异常，则记录详细异常信息
        logging.exception("视频处理过程中出现异常：{}".format(e))
    finally:
        # 无论是否异常，均记录视频处理结束的信息
        logging.info("视频处理结束。")


# 如果本文件作为主程序运行，则调用主函数启动程序
if __name__ == '__main__':
    main()
