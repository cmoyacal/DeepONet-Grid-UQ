import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Any

# compute metrics for N trajectories
def compute_metrics(s_true: list, s_pred: list, metrics: list, verbose: bool=False)-> list:
    out = []
    for m in metrics:
        temp = []
        for k in range(len(s_true)):
            temp.append(m(s_true[k], s_pred[k]))
        out.append(
            [
                np.round(100 * np.max(temp), decimals=5),
                np.round(100 * np.min(temp), decimals=5),
                np.round(100 * np.mean(temp), decimals=5),
            ]
        )
        del temp
    if verbose:
        try:
            print("l1-relative errors: max={:.3f}, min={:.3f}, mean={:.3f}".format(out[0][0], out[0][1], out[0][2]))
            print("l2-relative errors: max={:.3f}, min={:.3f}, mean={:.3f}".format(out[1][0], out[1][1], out[1][2]))
        except:
            print("not the correct metrics")
    return out

# l2 relative error
def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true) 

# l1 relative error
def l1_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_true - y_pred, ord=1) / np.linalg.norm(y_true, ord=1)

# test
def test(net: torch.nn, u: list, y: list) -> Any:
    mean = []
    std = []
    for input_k in zip(u, y):
        with torch.no_grad():
            mean_k, log_std_k = net(input_k)
            std_k = torch.exp(log_std_k)
        mean.append(mean_k.cpu().detach().numpy())
        std.append(std_k.cpu().detach().numpy())
    return mean, std

# test for sghmc
def test_sghmc(net: torch.nn, u: list, y: list) -> Any:
    pred = []
    for input_k in zip(u, y):
        with torch.no_grad():
            pred_k = net(input_k)
        pred.append(pred_k.cpu().detach().numpy())
    return pred

# test one trajectory
def test_one(net: torch.nn, u_k: torch.Tensor, y_k: torch.Tensor) -> Any:
    with torch.no_grad():
        mean, log_std = net((u_k,y_k))
        std = torch.exp(log_std)
    return mean.cpu().detach().numpy(), std.cpu().detach().numpy()

# update metrics
def update_metrics_history(history: dict, state: Any) -> dict:
    history["max"].append(state[0])
    history["min"].append(state[1])
    history["mean"].append(state[2])
    return history

# plot prediction and confidence interval
def plot_pred_UQ(sensors, u, y, s, s_mean, s_std, xlabel="$y$", ylabel="$s^\dagger(u)(y)$", size=10, v_lims=False):
    # plot prediction with confidence interval
    y = y.reshape(-1,)
    u = u.reshape(-1,)
    s = s.reshape(-1,)
    s_mean = s_mean.reshape(-1,)
    s_std = s_std.reshape(-1,)
    
    plt.figure()
    plt.plot(sensors, u, ":k", label="Input")
    plt.plot(y, s, "-b", label="True")
    plt.plot(y, s_mean, "--r", label="Mean prediction")
    plt.fill(
        np.concatenate([y, y[::-1]]),
        np.concatenate([s_mean - 1.9600 * s_std, (s_mean + 1.9600 * s_std)[::-1]]),
        alpha=.5, fc="c", ec="None", label="95% confidence interval"
    )
    if v_lims:
        t_clear, T = 1.0, 7.0
        xvals = [.33, .5, 1.5, T-t_clear]
        yvals = [.7, .8, .9, .95]

        plt.hlines(y=yvals[0], xmin=t_clear, xmax = t_clear + xvals[0], linewidth=2, color="black", linestyle="dashdot", label="UVLS")
        plt.vlines(x=t_clear + xvals[0], ymin=yvals[0], ymax=yvals[1], linewidth=2, color="black", linestyle="dashdot")
        plt.hlines(y=yvals[1], xmin=t_clear + xvals[0], xmax=t_clear + xvals[1], linewidth=2, color="black", linestyle="dashdot")
        plt.vlines(x=t_clear + xvals[1], ymin=yvals[1], ymax=yvals[2], linewidth=2, color="black", linestyle="dashdot")
        plt.hlines(y=yvals[2], xmin=t_clear + xvals[1], xmax=t_clear + xvals[2], linewidth=2, color="black", linestyle="dashdot")
        plt.vlines(x=t_clear + xvals[2], ymin=yvals[2], ymax=yvals[3], linewidth=2, color="black", linestyle="dashdot")
        plt.hlines(y=yvals[3], xmin=t_clear + xvals[2], xmax=t_clear + xvals[3], linewidth=2, color="black", linestyle="dashdot")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim((0.5,1.05))
    plt.legend(prop={'size':size})

# compute confidence interval
def fraction_in_CI(s, s_mean, s_std, xi=2, verbose=False):
    # fraction of true trajectory in predicted CI
    s = s.reshape(-1,)
    s_mean = s_mean.reshape(-1,)
    s_std = s_std.reshape(-1,)
    x = (np.abs(s - s_mean) <= xi * s_std)
    ratio = sum(x) / s.shape[0]
    if verbose:
        print("% of the true traj. within the error bars is {:.3f}".format(100 * ratio))
    return ratio

# trajectory relative error
def trajectory_rel_error(s_true, s_pred, verbose=False):
    l1_error = l1_relative_error(s_true.reshape(-1,), s_pred.reshape(-1,))
    l2_error = l2_relative_error(s_true.reshape(-1,), s_pred.reshape(-1,))
    if verbose:
        print("The L1relative error is {:.5f}:".format(l1_error))
        print("The L2relative error is {:.5f}:".format(l2_error))
    return l1_error, l2_error 

# initialize nn parameters
def init_params(net: torch.nn) -> list:
    list_pars = []
    for p in net.parameters():
        temp = torch.zeros_like(p.data)
        list_pars.append(temp.cpu().detach().numpy()) 
    return list_pars