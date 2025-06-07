# File: pll_nn/gradient_based_weighting.py

import torch
import torch.nn as nn

class PINNWeighting:
    """
    A class to dynamically adjust weights between different PINN loss terms
    (loss_data, loss_dt, loss_pinn, loss_pinn_ic).

    Args:
        model (nn.Module): The neural network model instance.
        cfg (OmegaConf): Configuration containing weighting settings.
        device (torch.device): Device to place weight tensors on (CPU or GPU).
        loss_dimension (int): Number of PDE residual terms (e.g., 1 or 2).
        wandb_run (wandb Run): Current WandB run for logging.
        beta (float): Exponential moving average factor for weights (if used).
    """
    def __init__(self, model, cfg, device, loss_dimension, wandb_run, beta=0.99):
        self.model = model
        self.weights = cfg.nn.weighting.weights  # [loss_data, loss_dt, loss_pinn, loss_pinn_ic]
        self.scheme = cfg.nn.weighting.update_weight_method  # "Static", "Gradient", "Ntk", or "Sam"
        self.update_weights_freq = cfg.nn.weighting.update_weights_freq
        self.beta = beta
        self.wandb_run = wandb_run
        # If time_factored_loss or flag_mean_weights is True, treat all PDE terms as a single dimension
        self.loss_dimension = (1 if (cfg.nn.weighting.flag_mean_weights or cfg.nn.time_factored_loss) else loss_dimension)
        self.device = device
        self.initialize_weights()
        self.epoch_flag = -1

    def initialize_weights(self):
        """
        Initialize the internal weight vector and any bookkeeping for the chosen scheme.
        """
        # Validate that initial weights list has exactly 4 positive values
        if self.weights is None or len(self.weights) != 4 or any(w < 0 for w in self.weights):
            raise ValueError("Weights must be a list of 4 non-negative values: [loss_data, loss_dt, loss_pinn, loss_pinn_ic]")

        # Build a tensor of weights with length = 1 + loss_dimension + loss_dimension + 1
        updated_weights = (
            [self.weights[0]] +                         # weight for loss_data
            [self.weights[1]] * self.loss_dimension +   # weight for each loss_dt term
            [self.weights[2]] * self.loss_dimension +   # weight for each loss_pinn term
            [self.weights[3]]                           # weight for loss_pinn_ic
        )
        self.weights = torch.tensor(updated_weights, device=self.device, dtype=torch.float32,
                                    requires_grad=(self.scheme == 'Sam'))

        # Build balancing factors (just for scaling, optional)
        balancing_term = ([1.0] +
                          [0.01] * self.loss_dimension +
                          [0.01] * self.loss_dimension +
                          [0.01])
        self.balancing_term = torch.tensor(balancing_term, device=self.device, dtype=torch.float32)

        # Build a mask: if an initial weight is zero, we never update that position
        self.weight_mask = torch.where(self.weights == 0,
                                       torch.tensor(0.0, device=self.device),
                                       torch.tensor(1.0, device=self.device))

        # If using SAM scheme, wrap weights as nn.Parameter
        if self.scheme == 'Sam':
            self.soft_adaptive_weights = nn.Parameter(self.weights.clone())
            self.weights = self.soft_adaptive_weights

        # Log the initial weights to WandB (epoch = 0)
        self.log_weights(epoch=0)

    def compute_weighted_loss(self, loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch):
        """
        Combine individual loss terms into a single weighted loss.

        Args:
            loss_data (torch.Tensor): Supervised data MSE loss.
            loss_dt (list[torch.Tensor]): List of PDE residual losses (one per dimension).
            loss_pinn (list[torch.Tensor]): List of PINN residual losses (one per dimension).
            loss_pinn_ic (torch.Tensor): Initial-condition loss.
            epoch (int): Current epoch index (for weight updates).

        Returns:
            (torch.Tensor, list[torch.Tensor]): Tuple of (total_loss, each_weighted_loss_terms).
        """
        # If only one PDE dimension, average them into a scalar
        if self.loss_dimension == 1:
            loss_dt = torch.mean(torch.stack(loss_dt))
            loss_pinn = torch.mean(torch.stack(loss_pinn))

        # Build weighted terms
        loss_data_   = self.weights[0] * loss_data * self.loss_dimension
        loss_dt_     = [self.weights[1+i] * (loss_dt[i] if self.loss_dimension > 1 else loss_dt)
                        for i in range(self.loss_dimension)]
        loss_pinn_   = [self.weights[1+self.loss_dimension+i] *
                        (loss_pinn[i] if self.loss_dimension > 1 else loss_pinn)
                        for i in range(self.loss_dimension)]
        loss_pinn_ic_= self.weights[-1] * loss_pinn_ic * self.loss_dimension

        individual_weighted_losses = [loss_data_] + loss_dt_ + loss_pinn_ + [loss_pinn_ic_]
        total_loss = torch.stack(individual_weighted_losses).sum()

        # Update weights if in a dynamic scheme and it is the correct epoch
        if (epoch + 1) % self.update_weights_freq == 0 and self.scheme != 'Static':
            if epoch != self.epoch_flag:
                self.update_weights(individual_weighted_losses, epoch)
                self.epoch_flag = epoch

        return total_loss, individual_weighted_losses

    def update_weights(self, losses, epoch):
        """
        Update the weights vector according to the chosen scheme ("Gradient", "Ntk", or "Sam").
        """
        if self.scheme == 'Gradient':
            # Compute gradient norms for each loss term
            grad_norms = []
            for idx, loss in enumerate(losses):
                if self.weight_mask[idx] == 0.0:
                    grad_norms.append(torch.tensor(0.0, device=self.device))
                    continue
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                # Compute L2 norm of all parameter gradients
                grads = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
                sum_norm = torch.stack([torch.norm(g) for g in grads]).sum()
                grad_norms.append(sum_norm)

            grad_norms = torch.stack(grad_norms)
            grad_norms = torch.nan_to_num(grad_norms, nan=0.0)
            avg_norm = torch.mean(grad_norms)
            new_weights = avg_norm / (grad_norms + 1e-8)

        elif self.scheme == 'Ntk':
            # Compute NTK trace for each loss: sum of squared gradients per term
            ntk_traces = []
            for idx, loss in enumerate(losses):
                if self.weight_mask[idx] == 0.0:
                    ntk_traces.append(torch.tensor(0.0, device=self.device))
                    continue
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                trace = torch.tensor(0.0, device=self.device)
                for p in self.model.parameters():
                    if p.grad is not None:
                        trace += torch.sum(p.grad ** 2)
                ntk_traces.append(trace)

            ntk_traces = torch.stack(ntk_traces)
            ntk_traces = torch.nan_to_num(ntk_traces, nan=0.0)
            avg_trace = torch.mean(ntk_traces)
            new_weights = avg_trace / (ntk_traces + 1e-8)

        elif self.scheme == 'Sam':
            # Sharpness-Aware Minimization style update
            epsilon = 1e-8
            with torch.no_grad():
                # Assume self.soft_adaptive_weights.grad 已经存在
                delta = self.weight_mask * 0.1 / (self.soft_adaptive_weights.grad + epsilon)
                self.soft_adaptive_weights += delta
                self.log_weights(epoch)
                self.soft_adaptive_weights.grad.zero_()
            return

        else:
            # Static scheme: do nothing
            return

        # Apply moving-average update: new = old * mask + balancing_term * new_weights
        self.weights = (self.weights * self.weight_mask +
                        self.balancing_term * new_weights.to(self.device)) * self.weight_mask

        self.log_weights(epoch)

    def log_weights(self, epoch):
        """
        Log each weight value (and, if SAM, its gradient) to WandB.
        """
        if self.wandb_run is None:
            return

        if self.scheme == 'Sam':
            # Log both parameter value and gradient
            for i in range(len(self.soft_adaptive_weights)):
                self.wandb_run.log({
                    f"sam_weight_{i}": self.soft_adaptive_weights[i].item(),
                    f"sam_weight_grad_{i}": self.soft_adaptive_weights.grad[i].item() if self.soft_adaptive_weights.grad is not None else 0.0,
                    "epoch": epoch
                })
        else:
            # Log static/dynamic weights only
            for i, w in enumerate(self.weights):
                self.wandb_run.log({f"weight_{i}": w.item(), "epoch": epoch})

    def log_losses(self, loss_list, epoch, prefix):
        """
        Log a list of loss values under different names to WandB for debugging.

        Args:
            loss_list (list[torch.Tensor]): Individual loss terms.
            epoch (int): Current epoch index.
            prefix (str): Prefix for log names, e.g. "train_loss" or "val_loss".
        """
        if self.wandb_run is None:
            return
        for i, loss in enumerate(loss_list):
            self.wandb_run.log({f"{prefix}_term_{i}": loss.item(), "epoch": epoch})