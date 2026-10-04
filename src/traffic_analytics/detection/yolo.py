from __future__ import annotations
from typing import Sequence

import numpy as np
from traffic_analytics.core.types import Detections
from ultralytics import YOLO


def resolve_device(device :str) -> str:
 """'auto' -> 'cuda' if a GPU is available, else 'cpu'."""
 if device !="auto":
     return device
 import torch
 
 return "cuda" if torch.cuda.is_available() else "cpu"

class YoloDetector:
    
    def __init__(
        self,
        weights: str ="yolo11m.pt",
        conf:float =0.25,
        iou: float = 0.7,
        imgsz: int =640,
        device:str ="auto",
        class_ids:Sequence[int]=(2,3,5,7)
    ):
        self.device=resolve_device(device)    
        self.model:YOLO=YOLO(weights)
        self.conf=conf
        self.iou=iou
        self.imgsz=imgsz
        self.class_ids=list(class_ids)
        self.names=self.model.names
    
    def detect(self, frame: np.ndarray) -> Detections:
        result=self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            verbose=False,            
        )[0]
        boxes=result.boxes
        return Detections(
            xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            confidence=boxes.conf.cpu().numpy().astype(np.float32),
            class_id=boxes.cls.cpu().numpy().astype(int),
        )
            
    