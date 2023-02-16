import torch

from typing import Any

device = None

# initializes GPU
def init_gpu(use_gpu: bool = True, gpu_id: int = 0, verbose: bool=True) -> None:
    """Initializes torch device."""
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        if verbose:
            print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU not detected. Defaulting to CPU.")

# save model
def save(net: torch.nn, optimizer: torch.optim, save_path: str) -> None:
    state = {'state_dict': net.state_dict(),'optimizer': optimizer.state_dict()}
    torch.save(state, save_path)

# restore model
def restore(restore_path: str) -> Any:
    return torch.load(restore_path)