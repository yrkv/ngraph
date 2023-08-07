import torch
import torch.nn as nn

""" m^{ab}_a, m^{ab}_b, m_{ab} = Message(h_a, h_b, h_{ab})
For some nodes $a,b$ takes in the representations $V_a, V_b, E_{ab}$ and
outputs messages for each of $V_a, V_b, E_{ab}$. Inputs are expected to be
already concatenated to shape (*, ch_V*2 + ch_E), and are outputted concatenated.
"""
class MessageFunction(nn.Module):
    def __init__(self, main, zero_last=False, zero_index=-1):
        super().__init__()

        self.main = main
        
        if zero_last:
            with torch.no_grad():
                self.main[zero_index].weight.fill_(0)
                self.main[zero_index].bias.fill_(0)
    
    def forward(self, x):
        return self.main(x)


def message_tiny(V:int, E:int, zero=True):
    return MessageFunction(nn.Sequential(
        nn.Linear(V*2+E, V*2+E),
    ), zero_last=zero)

def message_tiny_plus(V:int, E:int, zero=True):
    return MessageFunction(nn.Sequential(
        nn.Linear(V*2+E, V*2+E),
        nn.ReLU(),
    ), zero_last=zero, zero_index=-2)

def message_small(V:int, E:int, hidden:int, zero=True):
    return MessageFunction(nn.Sequential(
        nn.Linear(V*2+E, hidden),
        nn.ReLU(),
        nn.Linear(hidden, V*2+E),
    ), zero_last=zero)

def message_small_plus(V:int, E:int, hidden:int, zero=True):
    return MessageFunction(nn.Sequential(
        nn.Linear(V*2+E, hidden),
        nn.ReLU(),
        nn.Linear(hidden, V*2+E),
        nn.ReLU(),
    ), zero_last=zero, zero_index=-2)
