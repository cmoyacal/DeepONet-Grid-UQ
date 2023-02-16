import numpy as np

from torch.utils import data
from typing import Any

# Data generator
class data_generator(data.Dataset):
    def __init__(self, u, y, s):
        self.u, self.y, self.s = u, y, s
        self.len = self.u.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.u[idx, ...], self.y[idx, ...]), self.s[idx, ...]

# generate training data
def get_data(path: str, verbose: bool=False)-> list:
    with np.load(path) as data:
        u = data["u"]
        y = data["y"]
        s = data["s"]
        t_sim = data["t"]
    if verbose:
        print("Shapes are: u {}, y {}, and s {}".format(u.shape, y.shape, s.shape))
    return (u, y, s, t_sim)

# generate test trajectory data
def get_traj_data(path: str, verbose: bool=False)-> list:
    with np.load(path) as data:
        u = data["u"]
        y = data["y"]
        s = data["s"]
    if verbose:
        idx = 0
        print("Shapes are: u {}, y {}, and s {}".format(u[idx].shape, y[idx].shape, s[idx].shape))
    return (u, y, s)