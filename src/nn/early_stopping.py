# File: pll_nn/early_stopping.py

import os

class EarlyStopping:
    """
    A utility to stop training when validation loss does not improve
    for a given patience.
    """
    def __init__(self, patience=10, min_delta=1e-4, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0

    def step(self, val_loss):
        """
        Check if validation loss improved. If not, increase counter.
        If counter > patience, triggers early stop.

        Returns:
            bool: True if training should stop (early stopping triggered)
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
            return False