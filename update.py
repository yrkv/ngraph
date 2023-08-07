import torch
import torch.nn as nn


# TODO: currently identical to MessageFunction, do I need both?
class UpdateFunction(nn.Module):
    def __init__(self, main, zero_last=False, zero_index=-1):
        super().__init__()

        self.main = main
        
        if zero_last:
            with torch.no_grad():
                self.main[zero_index].weight.fill_(0)
                self.main[zero_index].bias.fill_(0)
    
    def forward(self, x):
        return self.main(x)


def update_tiny(V:int, zero=True):
    return UpdateFunction(nn.Sequential(
        nn.Linear(V*3, V),
    ), zero_last=zero)

def update_sigmoid_tiny(V:int, zero=True):
    return UpdateFunction(nn.Sequential(
        nn.Sigmoid(),
        nn.Linear(V*3, V),
    ), zero_last=zero)

def update_small(V:int, hidden:int, zero=True):
    return UpdateFunction(nn.Sequential(
        nn.Linear(V*3, hidden),
        nn.ReLU(),
        nn.Linear(hidden, V),
    ), zero_last=zero)
