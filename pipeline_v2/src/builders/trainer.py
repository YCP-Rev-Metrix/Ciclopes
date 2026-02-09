from __future__ import annotations
from typing import Any, Dict
from core.config import Config
from core.registry import register
from ultralytics import YOLO

class YoloSegTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.model.name)
        self.train_args = cfg.training
        self.data_yaml = cfg.data.path

    def train(self):
        self.model.train(self.data_yaml, **self.train_args)