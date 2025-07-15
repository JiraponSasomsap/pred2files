import numpy as np
from pathlib import Path

class BaseSave:
    def __init__(self,
                 save_as:str|Path, 
                 name:str,
                 model_name_stem:str,
                 max_buffer_len:int=1000,):
        self.name = name
        self.model_name_stem = model_name_stem
        self._save_as = Path(save_as)
        self.counter = 0 # Start from 0
        self.counter_checkpoint = 1
        self.data = []
        self.max_buffer_len = max_buffer_len  # Default buffer length
    
    def _gen_filename(self):
        '''{start-frame}-{end-frame}.{source}.{model}'''
        filename = f'{self.counter_checkpoint}-{self.counter}.{self.name}.{self.model_name_stem}'
        return filename
    
    def _gen_parent(self):
        self._save_as.mkdir(parents=True,exist_ok=True)
        return self._save_as