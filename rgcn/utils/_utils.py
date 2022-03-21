import torch

def move_input_tensors(func):
    def wrapper(self, *args, **kwargs):
        tensors = [t for t in args if isinstance(t, torch.Tensor)] + \
                  [t for t in kwargs.values() if isinstance(t, torch.Tensor)]

        # Find device
        device = None
        for t in tensors:
            if t.device != torch.device('cpu'):
                device = t.device
                break

        # Move tensors to device
        if device is not None:
            for t in tensors:
                if t.device != device:
                    t = t.to(device)

        self.tensors = func(self, *args, **kwargs)
        return self.tensors