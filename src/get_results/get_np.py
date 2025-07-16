from .base import BaseGetResult

class GetNPy(BaseGetResult):
    def __init__(self, pred_files, max_buffer_len):
        super().__init__(pred_files, max_buffer_len)

    def get_result_by_iframe(self, iframe:int):
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
    
class GetNPz(BaseGetResult):
    def __init__(self, pred_files, max_buffer_len):
        super().__init__(pred_files, max_buffer_len)

    def get_result_by_iframe(self, key, iframe:int):
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