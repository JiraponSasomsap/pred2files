import numpy as np
import torch
from pathlib import Path
import yaml

class SaveTensor:
    def __init__(self):
        self.save_type = torch.Tensor.__name__
        raise NotImplementedError('NotImplementedError')

class SaveNumpy:
    def __init__(self, save_as:str|Path=None):
        self.save_type = np.ndarray.__name__
        self.save_as = Path(save_as) if save_as else None
        self.counter = 0
        self._counter_checkpoint = self.counter
        self.data_npy = []

    def _gen_filename(self, cfg):
        model_path = Path(cfg['path']['model_path'])
        source = Path(cfg['path']['source_path'])
        filename = f'{self._counter_checkpoint}-{self.counter-1}.{source.stem}.{model_path.stem}'
        return filename
    
    def _gen_parent(self, cfg):
        source = Path(cfg['path']['source_path'])
        self.save_as = self.save_as if self.save_as else source
        p = self.save_as.parent / source.stem
        p.mkdir(exist_ok=True)
        return p

    def _npy(self, parent, cfg):
        np_data = np.array(self.data_npy, dtype=object)
        filename = self._gen_filename(cfg=cfg)
        file = parent / f"{filename}.npy"
        np.save(file, np_data)
        self.data_npy = [] # clear data
        self._counter_checkpoint = self.counter

    def npy(self, arr, cfg, max_buffer_len=1000):
        parent = self._gen_parent(cfg)
        parent_npy = parent / "results-npy"
        parent_npy.mkdir(exist_ok=True)
        frame_count = cfg['cam_prop']['frame_count']

        self.counter += 1

        self.data_npy.append(arr)

        if len(self.data_npy) == max_buffer_len:
            self._npy(parent_npy, cfg)

        if self.counter == frame_count:
            self._npy(parent_npy, cfg)

        return parent
    
    def npz(self, cfg, save_cfg=False, **kwds):
        ''' ### This feature is coming soon'''
        raise NotImplementedError('This feature is coming soon')
        # parent = self._gen_parent(cfg)
        # filename = self._gen_filename(cfg)
        # if save_cfg:
        #     self.save_config(cfg)
        # filesave = parent / "results-npz"
        # filesave.mkdir(exist_ok=True)
        # sv = filesave / f"{filename}.npz"
        # np.savez(sv, **kwds)
        # self.counter += 1
        # return sv

    def save_config(self, config, save_as:Path):
        save_config(save_as / "info.yaml", config)

def save_config(file, cfg):
    with open(file, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)