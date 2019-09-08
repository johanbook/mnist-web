"""
This module defines and loads the used model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    Flatten layer. This is useful for when transitioning from
    a convolutional layer to a linear layer.
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Lenet(nn.Module):
    """
    LeNet-5 model for image classification.
    """
    def __init__(self):
        super(Lenet, self).__init__()
        self.arch = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.MaxPool2d(2),
            # 4x4 image
            Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return F.log_softmax(self.arch(x), dim=1)


def load_model(path):
    """
    Load Lenet model from a state dict.

    Args:
        path (str): Relative path to file with state dict.

    Returns:
        torch.nn.Module: The pre-trained network.
    """
    model = Lenet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
