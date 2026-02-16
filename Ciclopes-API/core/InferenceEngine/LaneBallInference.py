from ultralytics import YOLO
import torch

class LaneBallInference:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def infer(self, image):
        return self.model.predict(image)