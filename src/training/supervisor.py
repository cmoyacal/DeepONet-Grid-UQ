import torch
import numpy as np

from torch import distributions
from tqdm.auto import trange
from typing import Any

import utils.pytorch_utils as ptu

from utils.utils import compute_metrics, init_params, test, test_sghmc, update_metrics_history

def probabilistic_train(
    model: torch.nn,
    dataset: Any,
    params: dict,
    scheduler_params: dict=None,
    verbose: bool=True,
    loss_history: list=None,
    test_data: list=None,
    logging_file: str="./output/best-model.pt",
    metrics: list=None,
    ) -> Any:
    
    ## step 1: unpack test data and losses
    u_test, y_test, s_test = test_data
    L1_history, L2_history = loss_history

    if verbose:
        print("\n***** Probabilistic Training for {} epochs and using {} data samples*****\n".format(params["epochs"], dataset.len))

    ## step 2: split the dataset
    n_train = int(0.9 * dataset.len)
    n_val = dataset.len - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    ## step 3: load the torch dataset
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch size"], shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=n_val, shuffle=True)

    ## step 4: build the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning rate"])
    if scheduler_params is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_params["patience"], verbose=verbose, factor=scheduler_params["factor"])
    else:
        scheduler = None

    ## step 5: define best values, logger and pbar
    best = {}
    best["prob loss"] = np.Inf

    # logger
    logger = {}
    logger["prob loss"] = []
    pbar = trange(params["epochs"])

    ## step 6: training loop
    for epoch in pbar:
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in trainloader:
            ## batch training
            
            # step a: forward pass
            mean_pred, log_std_pred = model(x_batch)

            # step b: compute loss
            dist = distributions.Normal(mean_pred, torch.exp(log_std_pred))
            loss = -dist.log_prob(y_batch).mean()

            # step c: compute gradients and backpropagate
            optimizer.zero_grad()
            loss.backward()

            # log batch loss
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy().squeeze()

        try:
            avg_epoch_loss = epoch_loss / len(trainloader)
        except ZeroDivisionError as e:
            print("error: ", e, "batch size larger than number of training examples")

        # log epoch loss
        logger["prob loss"].append(avg_epoch_loss)

        # scheduler
        if scheduler is not None:
            if epoch % params["eval every"] == 0:
                with torch.no_grad():
                    epoch_val_loss = 0
                    model.eval()

                    for x_val_batch, y_val_batch in valloader:
                        ## batch validation

                        # step a: forward pass without computing gradients
                        mean_val_pred, log_std_val_pred = model(x_val_batch)

                        # step b: compute validation loss
                        val_dist = distributions.Normal(mean_val_pred, torch.exp(log_std_val_pred))
                        val_loss = -val_dist.log_prob(y_val_batch).mean()
                        epoch_val_loss += val_loss.detach().cpu().numpy().squeeze()

                    try:
                        avg_epoch_val_loss = epoch_val_loss / len(valloader)
                    except ZeroDivisionError as e:
                        print("error: ", e, "batch size larger than number of training examples")    
                ## take a scheduler step
                scheduler.step(avg_epoch_val_loss)

        # testing
        pred, std = test(model, u_test, y_test)
        metrics_state = compute_metrics(s_test, pred, metrics, verbose=False)
        L1_history = update_metrics_history(L1_history, metrics_state[0])
        L2_history = update_metrics_history(L2_history, metrics_state[1])

        if epoch % params["print every"] == 0 or epoch + 1 == params["epochs"]:
            if avg_epoch_loss < best["prob loss"]:
                best["prob loss"] = avg_epoch_loss
                ptu.save(model, optimizer, save_path = logging_file)
            
            pbar.set_postfix(
                {
                    'Train-Loss': avg_epoch_loss,
                    'Best-Loss': best["prob loss"],
                    'L1-[max, min, mean]': metrics_state[0], 
                    'L2-[max, min, mean]': metrics_state[1],      
                 })
        del metrics_state

    return logger, (L1_history, L2_history)
    
# train sghmc    
def sghmc_train(
    model: torch.nn,
    dataset: Any,
    params: dict,
    sghmc_params: dict=None,
    test_data: list=None,
    loss_history: list=None,
    verbose: bool=True,
    device: torch.device=None,
    metrics: Any=None,
    ) -> Any:
    
    ## step 1: unpack test data and losses and initialize velocity params
    u_test, y_test, s_test = test_data
    L1_history, L2_history = loss_history
    N = params["N"]
    velocity = init_params(model)
    n_test_hist = params["size test history"]

    if verbose:
        print("\n***** SGHMC Training for {} epochs and using {} data samples*****\n".format(params["epochs"], dataset.len))

    ## step 2: load the torch dataset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch size"], shuffle=True)

    ## step 3: define best values, logger and pbar
    best = {}
    best["ge"] = np.Inf

    # logger
    logger = {}
    logger["pred test"] = []
    logger["ge"] = []
    pbar = trange(params["epochs"])

    ## step 4: training loop
    for epoch in pbar:
        epoch_loss = 0

        for x_batch, y_batch in trainloader:
            ## batch training
            
            # step a: forward pass
            model.zero_grad()
            ge_out = model(x_batch)

            # step b: compute loss
            loss = (N / params["batch size"]) * ((ge_out - y_batch) ** 2).sum()

            # step c: compute gradients and backpropagate
            loss.backward()
            if params["use grad norm"]:
                torch.nn.utils.clip_grad_value_(model.parameters(), params["grad norm"])

            # step d: update parameters
            for k, p in enumerate(model.parameters()):
                brownie = np.random.normal(0, 1, 1)[0]
                grad = p.grad.data.cpu().detach().numpy() / sghmc_params["sigma"]
                velocity[k] = - grad * sghmc_params["eta"] + (1 - sghmc_params["alpha"]) * velocity[k] + brownie * sghmc_params["self sigma"]
                p.data.add_(torch.tensor(velocity[k], requires_grad=False, device=device, dtype=torch.float32))

            # step e: log batch loss
            epoch_loss += loss.detach().cpu().numpy().squeeze()

        try:
            avg_epoch_loss = epoch_loss / len(trainloader)
        except ZeroDivisionError as e:
            print("error: ", e, "batch size larger than number of training examples")

        ## log average epoch loss
        logger["ge"].append(avg_epoch_loss)

        ## test
        pred = test_sghmc(model, u_test, y_test)
        metrics_state = compute_metrics(s_test, pred, metrics, verbose=False)
        L1_history = update_metrics_history(L1_history, metrics_state[0])
        L2_history = update_metrics_history(L2_history, metrics_state[1])

        ## update best and print
        if epoch % params["print every"] == 0 or epoch + 1 == params["epochs"]:
            best["ge"] = avg_epoch_loss if avg_epoch_loss < best["ge"] else best["ge"]
            
            pbar.set_postfix({
                'ge': np.round(avg_epoch_loss, 10),
                'best-ge': np.round(best["ge"], 10),
                'L1-[max, min, mean]': metrics_state[0], 
                'L2-[max, min, mean]': metrics_state[1]
                }
                )
        del metrics_state

        ## save after burn in
        if epoch > params["burn in"]:
            temp = test_sghmc(model, u_test[:n_test_hist], y_test[:n_test_hist])
            logger["pred test"].append(temp)
            del temp

    return logger, (L1_history, L2_history)