import numpy as np
import torch
import copy


class EarlyStopping:
    """
    Optimized: Keeps best weights in RAM. Saves to disk only when requested.
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_state = None  # Store weights in RAM

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.update_best_state(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:  # Reduce print clutter too
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_best_state(val_loss, model)
            self.counter = 0

    def update_best_state(self, val_loss, model):
        '''Updates best state in memory without writing to disk.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Keeping weights in RAM.')
        self.val_loss_min = val_loss
        # Deep copy ensures we don't just point to the changing model
        self.best_state = copy.deepcopy(model.state_dict())

    def save_to_disk(self, path):
        '''Writes the best cached state to disk.'''
        if self.best_state is None:
            print("Warning: No best model state to save.")
            return
        torch.save(self.best_state, path)
        print(f"Best model saved to {path}")
