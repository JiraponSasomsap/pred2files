from ultralytics import YOLO
from pathlib import Path

class yolo(YOLO):
    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        self._model = Path(model)
        self._config = {
            'model': self._model.as_posix(),
            'model_name': self._model.name,
        }
        super().__init__(model, task, verbose)

    def set_config(self, 
                   source:str|Path):
        self._source = Path(source)
        cfg = {
            'source':self._source.as_posix(),
            'source_name':self._source.name,
        }
        self._config.update(cfg)
        return self
    
    def predict_and_save(self, source = None, stream = False, predictor=None, **kwargs):
        self._config['predict_params'] = kwargs
        return super().predict(source, stream, predictor, **kwargs)
