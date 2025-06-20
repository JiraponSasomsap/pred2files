from ultralytics import YOLO
from pathlib import Path

class yolo(YOLO):
    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        self._predictor_params = None
        super().__init__(model, task, verbose)
    
    def predict_and_save(self, source = None, stream = False, predictor=None, **kwargs):
        self._predictor_params = kwargs
        return super().predict(source, stream, predictor, **kwargs)
