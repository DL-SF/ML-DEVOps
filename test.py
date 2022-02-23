import torch
import torch.nn as nn
import torchvision
from torch.autograd.gradcheck import gradcheck

def test_sanity(self):
    input = (Variable(torch.randn(20, 20).double(), requires_grad=True), )
    model = nn.Linear(20, 1).double()
    test = gradcheck(model, input, eps=1e-6, atol=1e-4)
    print(test)
