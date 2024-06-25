import torch 

class CyclicReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):

    def __init__(self, first_lr: float, last_lr: float, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.first_lr = first_lr
        self.last_lr = last_lr
        self.dflt_state = self.state_dict()
        self._reset_state()

    def step(self, *args, **kwargs):
        
        super().step(*args, **kwargs)

        if self._is_threshold_reached():
            self._reset_state()

    def _is_threshold_reached(self) -> bool:
        
        for param_group in self.optimizer.param_groups:
            old_lr = float(param_group["lr"])
            if old_lr < self.last_lr:
                return True 
        
        return False

    def _update_lr(self, lr: float):

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr 

    def _reset_state(self):

        self.load_state_dict(self.dflt_state)
        self._update_lr(self.first_lr)
