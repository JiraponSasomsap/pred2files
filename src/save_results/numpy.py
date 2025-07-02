import numpy as np
from pathlib import Path
import yaml

from .base import BaseSave
    
class SaveNumpy(BaseSave):
    def __init__(self, save_as = None, max_buffer_len: int = 1000):
        super().__init__(save_as, max_buffer_len)
        self.save_type = np.ndarray.__name__

    def dump_npy(self, parent=None):
        """Dump the data to a .npy file."""
        if len(self.data) == 0:
            return
        
        if parent is None:
            parent = self._gen_parent() / "results-npy"
            parent.mkdir(exist_ok=True, parents=True)

        np_data = np.array(self.data, dtype=object)
        filename = self._gen_filename()
        file = parent / f"{filename}.npy"
        
        np.save(file, np_data)

        self.data = [] # clear data
        self.counter_checkpoint = self.counter+1

    def npy(self, arr):
        parent = self._gen_parent()
        parent_npy = parent / "results-npy"
        parent_npy.mkdir(exist_ok=True)

        self.data.append(arr)
        self.counter += 1
        
        if len(self.data) == self.max_buffer_len:
            self.dump_npy(parent_npy)

        return parent