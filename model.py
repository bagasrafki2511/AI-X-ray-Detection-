# Sistem/otak dari Ai
import torch
import torch.nn as nn
from torchvision import models

class XRayModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.efficientnet_v2_s(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model():
    model = XRayModel()
    model.eval()
    return model
