import numpy as np
from pathlib import Path
import yaml

from .base import BaseSave
    
class SaveNumpy(BaseSave):
    def __init__(self, save_as = None):
        super().__init__(save_as)
        self.save_type = np.ndarray.__name__

    def _npy(self, parent):
        np_data = np.array(self.data, dtype=object)
        filename = self._gen_filename()
        file = parent / f"{filename}.npy"
        np.save(file, np_data)
        self.data = [] # clear data
        self._counter_checkpoint = self.counter

    def npy(self, arr, max_buffer_len=1000):
        parent = self._gen_parent()
        parent_npy = parent / "results-npy"
        parent_npy.mkdir(exist_ok=True)
        frame_count = self.config['cam_prop']['frame_count']

        self.counter += 1
        self.data.append(arr)

        if len(self.data) == max_buffer_len:
            self._npy(parent_npy)
            return parent

        if self.counter == frame_count:
            self._npy(parent_npy)

        return parent