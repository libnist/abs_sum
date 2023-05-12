import torch

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from torch.optim.lr_scheduler import LambdaLR


def Accuracy(ignore_index=None):

    def accuracy(y_pred: torch.tensor,
                 y_true: torch.tensor):
        
        if ignore_index is not None:  
            match_ = y_pred == y_true
            
            mask = y_true != ignore_index        

            match_ = match_ & mask
            return torch.sum(match_) / torch.sum(mask)
        else:
            match_ = y_pred == y_true
            return torch.sum(match_) / torch.numel(match_)
    return accuracy

def summary_writer(path: str,
                   model_name: str,
                   time: str = None,
                   extra: str = None):
    if not time:
        time = datetime.now().strftime("%Y-%m-%d")
    
    if not extra:
        log_dir = os.path.join(path, model_name, time)
    else:
        log_dir = os.path.join(path, model_name, time, extra)
        
    return SummaryWriter(log_dir=log_dir)



def get_transformer_scheduler(optimizer: torch.optim.Optimizer,
                              warmup_steps: int,
                              d_model: int,
                              last_epoch: int = -1):

    warmup_coeff = warmup_steps**-1.5

    # Inverse of the optimizers default lr is used to neutrize the effect of it.
    d_model_coeff = (d_model**-0.5) * (1 / optimizer.param_groups[0]["lr"])

    def lr_lambda(current_step):
        current_step += 1
        return d_model_coeff * min(current_step**-0.5,
                                   current_step * warmup_coeff)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_modified_transformer_scheduler(optimizer: torch.optim.Optimizer,
                                       warmup_steps: int,
                                       coeff: float = 0.002,
                                       last_epoch: int = -1):

    warmup_coeff = warmup_steps**-1.5

    # Inverse of the optimizers default lr is used to neutrize the effect of it.
    coeff = coeff * (1 / optimizer.param_groups[0]["lr"])

    def lr_lambda(current_step):
        current_step += 1
        return coeff * min(current_step**-0.5, current_step * warmup_coeff)

    return LambdaLR(optimizer, lr_lambda, last_epoch)