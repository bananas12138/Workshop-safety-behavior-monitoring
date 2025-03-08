import torch.hub

model = torch.hub.load('ultralytics/yolov5', 'custom', path='helmet_head_person_l.pt')
print(model)  # 输出模型结构
print(torch.hub.list('ultralytics/yolov5'))  # 应输出 ['yolov5s', 'yolov5m', ...]
