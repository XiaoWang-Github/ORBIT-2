import torch
from torch import Tensor

class LogTransform(torch.nn.Module):
    """Log-transform a tensor image.
    This transform does not support PIL Image.
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:

    """

    def __init__(self, m2mm=True, LOG1P=True, thres_mm_per_day=0.25):
        super().__init__()
        self.epsilon = torch.finfo(float).eps
        self.m2mm = m2mm
        self.LOG1P = LOG1P
        self.thres_mm_per_day = thres_mm_per_day # 0.1 inch = 0.25 mm / day

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
            assume [m] in total daily precipitation

        Returns:
            Tensor: Normalized Tensor image.
        """
        
        if self.m2mm:
            tensor *= 1000.
            tensor = torch.where(tensor <= self.thres_mm_per_day, torch.tensor(0), tensor) # suppress 0.25mm/day to 0
        else:
            # in case unit is [m] in trianing
            thres_mm_per_day = self.thres_mm_per_day / 1000.
            tensor = torch.where(tensor <= thres_mm_per_day, torch.tensor(0), tensor) # suppress 0.00025m/day to 0
        
        if self.LOG1P:
            return torch.log1p(tensor)
        else:
            return torch.log(tensor + self.epsilon)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(M2mm={self.m2mm}, Log(x+1)={self.LOG1P})"
