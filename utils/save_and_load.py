# Import libs
import torch
import pathlib

from pathlib import Path


def save_model(model: torch.nn.Module,
               path: pathlib.Path,
               name: str,
               optimizer: torch.optim.Optimizer = None,
               lr_scheduler = None) -> None:
    """Saves a given model into the given path.

    Args:
        model (torch.nn.Module): A PyTorch model.
        path (pathlib.Path): A path to save the model in.
        name (str): The name under which we'll be saving the model.
        optimizer (torch.optim.Optimzier): A torch optimizer.
        lr_scheduler: Learning rate scheduler.
    """
    path = path / name
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    model_path = path /  "model.pth"
    optimizer_path = path / "optim.pth"
    lr_scheduler_path = path / "lr_scheduler.pth"
    
    torch.save(model.state_dict(), model_path)
    print(f"\tModel is saved in: {model_path}")
    
    if optimizer:
        torch.save(optimizer.state_dict(), optimizer_path)
        print(f"\tModels optimizer is saved in: {optimizer_path}")
        if lr_scheduler:
            torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
            print(f"\tOptimizers lr_scheduler is saved in: {lr_scheduler_path}")
            
def load_model(model: torch.nn.Module,
               path: pathlib.Path,
               name: str,
               device: str,
               optimizer: torch.optim.Optimizer = None,
               lr_scheduler = None):
    """Loads and returns a presaved model.

    Args:
        model (torch.nn.Module): A model to load weights in.
        path (pathlib.Path): The path in which a saved model exists.
        name (str): The name under which the model is saved.
        optimizer (torch.optim.Optimizer, optional): Optimizer.
        Defaults to None.
        lr_scheduler (_type_, optional): Learning rate scheduler.
        Defaults to None.
    """
    path = path / name
    model_path = path / "model.pth"
    optimizer_path = path / "optim.pth"
    lr_scheduler_path = path / "lr_scheduler.pth"
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"\tModel is loaded from: {model_path}")
    
    if optimizer:
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"\tOptimizer is loaded from: {optimizer_path}")
        if lr_scheduler:
            lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))
            print(f"\tlr_scheduler is loaded from: {lr_scheduler_path}")
