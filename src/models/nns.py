import torch
import torch.nn as nn

from typing import Any

# MLP
class MLP(nn.Module):
    def __init__(self, layer_size: list, activation: str) -> None:
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        for k in range(len(layer_size) - 2):
            self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
            self.net.append(get_activation(activation))
        self.net.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
        self.net.apply(self._init_weights)
    
    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for k in range(len(self.net)):
            y = self.net[k](y)
        return y
    
# modified MLP
class modified_MLP(nn.Module):
    def __init__(self, layer_size: list, activation: str) -> None:
        super(modified_MLP, self).__init__()
        self.net = nn.ModuleList()
        self.U = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
        self.V = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
        self.activation = get_activation(activation)
        for k in range(len(layer_size) - 1):
            self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
        self.net.apply(self._init_weights)
        self.U.apply(self._init_weights)
        self.V.apply(self._init_weights)
            
    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.activation(self.U(x))
        v = self.activation(self.V(x))
        for k in range(len(self.net) - 1):
            y = self.net[k](x)
            y = self.activation(y)
            x = y * u + (1 - y) * v
        y = self.net[-1](x)
        return y

# get activation function from str
def get_activation(identifier: str) -> Any:
    """get activation function."""
    return{
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sin": sin_act(),
            "softplus": nn.Softplus(),
            "Rrelu": nn.RReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "Mish": nn.Mish(),
    }[identifier]

# sin activation function
class sin_act(nn.Module):
    def __init__(self):
        super(sin_act, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

# vanilla DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch: dict, trunk: dict, use_bias: bool=True) -> None:
        super(DeepONet, self).__init__()
        if branch["type"] == "MLP":
            self.branch = MLP(branch["layer_size"], branch["activation"])
        elif branch["type"] == "modified":
            self.branch = modified_MLP(branch["layer_size"], branch["activation"])
        # trunk
        if trunk["type"] == "MLP":
            self.trunk = MLP(trunk["layer_size"], trunk["activation"])
        elif trunk["type"] == "modified":
            self.trunk = modified_MLP(trunk["layer_size"], trunk["activation"])

        self.use_bias  = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True) 

    def forward(self, x: list) -> torch.Tensor:
        u, y = x
        B = self.branch(u)
        T = self.trunk(y)
        
        s = torch.einsum("bi, bi -> b", B, T)
        s = torch.unsqueeze(s, dim=-1)
        
        return s + self.tau if self.use_bias else s

# probabilistic DeepONet
class prob_DeepONet(nn.Module):
    def __init__(self, branch: dict, trunk: dict, use_bias: bool=True) -> None:
        super(prob_DeepONet, self).__init__()
        if branch["type"] == "MLP":
            self.branch = MLP(branch["layer_size"][:-2], branch["activation"])
        elif branch["type"] == "modified":
            self.branch = modified_MLP(branch["layer_size"][:-2], branch["activation"])
        # trunk
        if trunk["type"] == "MLP":
            self.trunk = MLP(trunk["layer_size"][:-2], trunk["activation"])
        elif trunk["type"] == "modified":
            self.trunk = modified_MLP(trunk["layer_size"][:-2], trunk["activation"])

        self.use_bias  = use_bias

        if use_bias: 
            self.bias_mu = nn.Parameter(torch.rand(1), requires_grad=True)
            self.bias_std = nn.Parameter(torch.rand(1), requires_grad=True)

        self.branch_mu = nn.Sequential(
            get_activation(branch["activation"]), 
            nn.Linear(branch["layer_size"][-3], branch["layer_size"][-2], bias=True),
            get_activation(branch["activation"]),
            nn.Linear(branch["layer_size"][-2], branch["layer_size"][-1], bias=True)
            )
        
        self.branch_std = nn.Sequential(
            get_activation(branch["activation"]), 
            nn.Linear(branch["layer_size"][-3], branch["layer_size"][-2], bias=True),
            get_activation(branch["activation"]),
            nn.Linear(branch["layer_size"][-2], branch["layer_size"][-1], bias=True)
            )

        self.trunk_mu = nn.Sequential(
            get_activation(trunk["activation"]), 
            nn.Linear(trunk["layer_size"][-3], trunk["layer_size"][-2], bias=True),
            get_activation(trunk["activation"]),
            nn.Linear(trunk["layer_size"][-2], trunk["layer_size"][-1], bias=True)
            )
        
        self.trunk_std = nn.Sequential(
            get_activation(trunk["activation"]), 
            nn.Linear(trunk["layer_size"][-3], trunk["layer_size"][-2], bias=True),
            get_activation(trunk["activation"]),
            nn.Linear(trunk["layer_size"][-2], trunk["layer_size"][-1], bias=True)
            )

        self.branch_mu.apply(self._init_weights)
        self.branch_std.apply(self._init_weights)
        self.trunk_mu.apply(self._init_weights)
        self.trunk_std.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

    def forward(self, x: list) -> list:
        u, y = x
        b = self.branch(u)
        t = self.trunk(y)
        # branch prediction and UQ
        b_mu = self.branch_mu(b)
        b_std = self.branch_std(b)
        # trunk prediction and UQ
        t_mu = self.trunk_mu(t)
        t_std = self.trunk_std(t)

        # dot product
        mu = torch.einsum("bi, bi -> b", b_mu, t_mu)
        mu = torch.unsqueeze(mu, dim=-1)            
            
        log_std = torch.einsum("bi, bi -> b", b_std, t_std)
        log_std = torch.unsqueeze(log_std, dim=-1)
        if self.use_bias:
            mu += self.bias_mu
            log_std += self.bias_std
        return (mu, log_std)