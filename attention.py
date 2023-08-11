import torch
import torch.nn as nn



class AttentionFunction(nn.Module):
    def __init__(self, main, zero_last=False, zero_index=-1):
        super().__init__()

        self.main = main
        
        if zero_last:
            with torch.no_grad():
                self.main[zero_index].weight.fill_(0)
                self.main[zero_index].bias.fill_(0)
    
    def forward(self, x):
        return self.main(x)
    
def attention_tiny(V:int, K:int, zero=True):
    return AttentionFunction(nn.Sequential(
        nn.Linear(V, K*4),
    ), zero_last=zero)

def attention_tiny_plus(V:int, K:int, zero=True):
    return AttentionFunction(nn.Sequential(
        nn.Linear(V, K*4),
        nn.ReLU(),
    ), zero_last=zero, zero_index=-2)

def attention_small(V:int, K:int, hidden:int, zero=True):
    return AttentionFunction(nn.Sequential(
        nn.Linear(V, hidden),
        nn.ReLU(),
        nn.Linear(hidden, K*4),
    ), zero_last=zero)

def attention_small_plus(V:int, K:int, hidden:int, zero=True):
    return AttentionFunction(nn.Sequential(
        nn.Linear(V, hidden),
        nn.ReLU(),
        nn.Linear(hidden, K*4),
        nn.ReLU(),
    ), zero_last=zero, zero_index=-2)
