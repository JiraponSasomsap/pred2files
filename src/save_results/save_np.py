import numpy as np
from pathlib import Path
import yaml

from .base import BaseNumpySaveFile

class SaveFileNPy(BaseNumpySaveFile):
    def __init__(self, 
                 save_as, 
                 name, 
                 model_name_stem, 
                 project="results-npy",
                 max_buffer_len = 1000):
        super().__init__(save_as, name, model_name_stem, max_buffer_len)
        self.project = project

    def dump_npy(self, parent=None):
        """Dump the data to a .npy file."""
        if len(self.data) == 0:
            return
        
        if parent is None:
            parent = self._gen_parent() / self.project
            parent.mkdir(exist_ok=True, parents=True)

        np_data = np.array(self.data, dtype=object)
        filename = self._gen_filename()
        file = parent / f"{filename}.npy"
        
        np.save(file, np_data)

        self.data = [] # clear data
        self.counter_checkpoint = self.counter+1

    def save_npy(self, arr):
        parent = self._gen_parent()
        parent_npy = parent / self.project
        parent_npy.mkdir(exist_ok=True)

        self.data.append(arr)
        self.counter += 1
        
        if len(self.data) == self.max_buffer_len:
            self.dump_npy(parent_npy)

        return parent
    
class SaveFileNPz(BaseNumpySaveFile):
    def __init__(self, 
                save_as, 
                name, 
                model_name_stem, 
                project="results-npz",
                max_buffer_len = 1000):
        super().__init__(save_as, name, model_name_stem, max_buffer_len)
        self.project = project
    
    def dump_npz(self, parent=None):
        """Dump the data to a .npz file."""
        if len(self.data) == 0:
            return
        
        if parent is None:
            parent = self._gen_parent() / self.project
            parent.mkdir(exist_ok=True, parents=True)

        filename = self._gen_filename()
        file = parent / f"{filename}.npz"
        
        df = {}

        for data in self.data:
            for key, val in data.items():
                if key not in df:
                    df[key] = [val]
                else:
                    df[key].append(val)

        for key in df:
            df[key] = np.array(df[key], dtype=object)

        np.savez(file, **df)

        self.data = [] # clear data
        self.counter_checkpoint = self.counter+1

    
    def save_npz(self, data_dict:dict):
        parent = self._gen_parent()
        parent_npy = parent / self.project
        parent_npy.mkdir(exist_ok=True)

        self.data.append(data_dict)
        self.counter += 1
        
        if len(self.data) == self.max_buffer_len:
            self.dump_npz(parent_npy)

        return parent