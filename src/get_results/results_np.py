from pathlib import Path
import numpy as np

class GetResultsNumpy:
    def __init__(self, pred_files, frame_gap):
        self.frame_gap = frame_gap
        self.pred_files = Path(pred_files)
        self.map = []
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
            self.map.append([[start, end], file])

    def _load_results(self, file):
        print(self.map_checkpoint)
        return np.load(file, allow_pickle=True)
    
    def _get_mapped(self, iframe):
        # if iframe not in range(self.start_frame, self.end_frame+1):
        #     print(self.map)
        #     raise ValueError(f"iframe {iframe} not mapped in any range.")
        for m in self.map:
            if iframe in range(m[0][0], m[0][1]):
                self.map_checkpoint = m
                self.data_checkpoint = self._load_results(m[1])
                return m
    
    def get_results(self, iframe:int):
        iframe = iframe-1
        if self.map_checkpoint is None:
            self._get_mapped(iframe)
        else:
            if iframe in range(self.map_checkpoint[0][0], 
                               self.map_checkpoint[0][1]):
                # index = iframe - self.map_checkpoint[0][0]
                index = iframe % self.frame_gap
                return self.data_checkpoint[index]
            else: self._get_mapped(iframe)
        # index = iframe - self.map_checkpoint[0][0]
        index = iframe % self.frame_gap
        return self.data_checkpoint[index]

    def get_results_conv(self, iframe , frame):
        h, w = frame.shape[:2]
        boxes = [[box[0]*w, box[1]*h, box[2]*w, box[3]*h] if len(box) > 0 else [] for box in self.get_results(iframe=iframe)]
        return boxes