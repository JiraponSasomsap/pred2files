from pathlib import Path
import numpy as np
from dataclasses import dataclass

@dataclass
class NPYResult:
    bounding_boxes = None
    bounding_boxes_norm = None
    center = None
    center_norm = None


class GetResultsNumpy:
    def __init__(self, pred_files, frame_gap):
        self.frame_gap = frame_gap
        self.pred_files = Path(pred_files)
        self.map = []
        self.start_frame = np.inf
        self.end_frame = np.inf * -1
        self.map_checkpoint = None
        self.data_checkpoint = None
        self._data = NPYResult()

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
            
    def get_raw_results(self, iframe:int):
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
    
    def get_results(self, iframe, frame) -> NPYResult:
        h,w = frame.shape[:2]
        bboxes = []
        bboxes_norm = []
        center = []
        center_norm = []

        for box in self.get_raw_results(iframe=iframe):
            if len(box) == 0:
                bboxes.append(np.empty((0 ,4)))
                bboxes_norm.append(np.empty((0 ,4)))
                center.append(np.empty(0, 2))
                center_norm.append(np.empty(0, 2))
            elif len(box) == 4:
                # center
                b1, b2 = box.reshape(2, 2)
                ct = ((b2 - b1) / 2) + b1
                center_norm.append(ct)
                center.append(ct * [w,h])
                
                # bounding boxes
                bboxes_norm.append(np.array(box, dtype=np.float32))
                box = [box[0]*w, box[1]*h, box[2]*w, box[3]*h]
                bboxes.append(np.array(box, dtype=np.float32))
            else:
                raise ValueError(f"Invalid bounding box length: {len(box)} → {box}")
        
        self._data.bounding_boxes = bboxes
        self._data.bounding_boxes_norm = bboxes_norm
        self._data.center = center
        self._data.center_norm = center_norm
        return self._data