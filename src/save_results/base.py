import numpy as np
from pathlib import Path
import yaml

class BaseSave:
    def __init__(self, save_as:str|Path=None, max_buffer_len:int=1000):
        self.save_type = None
        self.save_as = Path(save_as) if save_as else None
        self.counter = 0 # Start from 0
        self.counter_checkpoint = 1
        self.data = []
        self.config = {}
        self.max_buffer_len = max_buffer_len  # Default buffer length

    def set_config(self, model_path:Path, video_path:Path, fps:int, max_frame_count:int):
        self.config = {
            'path':{
                'model_path':model_path.as_posix(),
                'source_path':video_path.as_posix(),
            },
            'model':model_path.name,
            'source':video_path.name,
            'save_type':self.save_type,
            'cam_prop':{
                'fps':fps,
                'frame_count': max_frame_count
            },
            'predictor':{
                'names':model_path.stem
            }
        }
    
    def _gen_filename(self):
        '''{start-frame}-{end-frame}.{source}.{model}'''
        model_path = Path(self.config['path']['model_path'])
        source = Path(self.config['path']['source_path'])
        filename = f'{self.counter_checkpoint}-{self.counter}.{source.stem}.{model_path.stem}'
        return filename
    
    def _gen_parent(self):
        source = Path(self.config['path']['source_path'])
        self.save_as = self.save_as if self.save_as else source
        parent_path = self.save_as.parent / source.stem
        parent_path.mkdir(exist_ok=True)
        return parent_path
    
    def save_config(self, save_as:Path):
        file = save_as / "info.yaml"
        with open(file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        return self