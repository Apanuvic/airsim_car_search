import torch
import cv2

class CarDetector:
    def __init__(self, model_name='yolov5s', device='cpu'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(device)
        self.car_ids = [2]  # класс 'car'

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb)
        cars = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) in self.car_ids:
                x1, y1, x2, y2 = map(int, box)
                cars.append((x1, y1, x2, y2, float(conf)))
        return cars
