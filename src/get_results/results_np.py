from pathlib import Path
import numpy as np

class GetResultsNumpy:
    def __init__(self, pred_files, max_buffer_len):
        self.max_buffer_len = max_buffer_len
        self.pred_files = Path(pred_files)
        self.mapper = []

        self.start_frame = np.inf
        self.end_frame = np.inf * -1

        self.map_checkpoint = None
        self.data_checkpoint = None

        for file in self.pred_files.glob('*'):
            frame_len = file.stem.split('.')[0]
            start, end = int(frame_len.split('-')[0]), int(frame_len.split('-')[1])
            if start < self.start_frame:
                self.start_frame = start
            if end > self.end_frame:
                self.end_frame = end
            self.mapper.append([[start, end], file])

    def _load_results(self, file):
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        frame_range, npy_path = self.map_checkpoint

        print(f"{YELLOW}Frame Range : {frame_range[0]} to {frame_range[1]}{RESET}")
        print(f"{YELLOW}Numpy Path  : {npy_path.as_posix()}{RESET}")
        return np.load(file, allow_pickle=True)
    
    def _get_mapped(self, iframe):            
        for mp in self.mapper:
            if iframe in range(mp[0][0], mp[0][1]+1):
                self.map_checkpoint = mp
                self.data_checkpoint = self._load_results(mp[1])
                return mp
        return None
    
    def get_npz(self, key, iframe:int):
        mapper = None
        if self.map_checkpoint is None:
            mapper = self._get_mapped(iframe)
        else:
            if iframe in range(self.map_checkpoint[0][0], 
                               self.map_checkpoint[0][1]+1):
                index = (iframe-1) % self.max_buffer_len
                return self.data_checkpoint[key][index]
            else: 
                mapper = self._get_mapped(iframe)

        if mapper is None:
            raise ValueError(f"Frame {iframe} not found in the results.")

        index = (iframe-1) % self.max_buffer_len
        return self.data_checkpoint[key][index]
    
    def get_npy(self, iframe:int):
        mapper = None
        if self.map_checkpoint is None:
            mapper = self._get_mapped(iframe)
        else:
            if iframe in range(self.map_checkpoint[0][0], 
                               self.map_checkpoint[0][1]+1):
                index = (iframe-1) % self.max_buffer_len
                return self.data_checkpoint[index]
            else: 
                mapper = self._get_mapped(iframe)

        if mapper is None:
            raise ValueError(f"Frame {iframe} not found in the results.")

        index = (iframe-1) % self.max_buffer_len
        return self.data_checkpoint[index]