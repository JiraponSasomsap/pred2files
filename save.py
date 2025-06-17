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

    def _gen_filename(self, cfg):
        model_path = Path(cfg['model'])
        source = Path(cfg['source'])
        filename = f'{self.counter}__source__{source.stem}__model__{model_path.stem}'
        return filename
    
    def _gen_parent(self, cfg):
        source = Path(cfg['source'])
        self.save_as = self.save_as if self.save_as else source
        p = self.save_as.parent / source.stem
        p.mkdir(exist_ok=True)
        return p

    def npy(self, arr, cfg, save_cfg=False):
        self.counter += 1
        parent = self._gen_parent(cfg)
        filename = self._gen_filename(cfg)
        if save_cfg:
            self.save_config(cfg)
        filesave = parent / "results"
        filesave.mkdir(exist_ok=True)
        sv = filesave / f"{filename}.npy"
        np.save(sv, arr)
        return sv
    
    def npz(self, cfg, save_cfg=False, **kwds):
        self.counter += 1
        parent = self._gen_parent(cfg)
        filename = self._gen_filename(cfg)
        if save_cfg:
            self.save_config(cfg)
        filesave = parent / "results"
        filesave.mkdir(exist_ok=True)
        sv = filesave / f"{filename}.npz"
        np.savez(sv, **kwds)
        return sv

    def save_config(self, config, save_as=None):
        if save_as is None:
            fileconfig = self._gen_parent(config) / "config"
            fileconfig.mkdir(exist_ok=True)
            fileconfig = fileconfig / f"{self._gen_filename(config)}.yaml"
            save_config(fileconfig, config)
        else:
            save_as = Path(save_as)
            save_config(save_as / "config.yaml", config)

def save_config(file, cfg):
    if 'source_name' not in list(cfg.keys()):
        raise ValueError("'source_name' not found in config. Please run set_source() before saving.")
    
    with open(file, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)