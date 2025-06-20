import torch

class SaveTensor:
    def __init__(self):
        self.save_type = torch.Tensor.__name__
        raise NotImplementedError('NotImplementedError')