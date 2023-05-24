import torch
from torch import nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
  def __init__(self, t, alpha):
    super().__init__()
    self.alpha = alpha
    self.t = t
    self.kldiv = nn.KLDivLoss(reduction="batchmean")

  def forward(self, y_pred, y_true):
    y_pred = F.log_softmax(y_pred / self.t, dim=-1)
    y_true = F.softmax(y_true / self.t, dim=-1)
    return self. alpha * self.kldiv(y_pred, y_true) * (self.t ** 2) / y_pred.shape[0]

class ZeroLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, y_pred, y_true):
        return torch.tensor(0.0, device=self.device)
      
class HardLoss(nn.Module):
  def __init__(self, alpha, pad_id):
    super().__init__()
    self.alpha = alpha
    self.ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    
  def forward(self, y_pred, y_true):
    return self.alpha * self.ce(y_pred.reshape(-1, y_pred.shape[-1]),
                                y_true.reshape(-1))
    
class FeatureLoss(nn.Module):
  def __init__(self, alpha, teacher_dim, student_dim):
    super().__init__()
    self.alpha = alpha
    self.mse = nn.MSELoss()
    self.linear = nn.Linear(teacher_dim, student_dim)

  def forward(self, y_pred, y_true):
    return self.alpha * self.mse(y_pred, self.linear(y_true))