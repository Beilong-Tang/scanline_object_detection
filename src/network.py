import torch
from torchvision.models import mobilenet_v2, resnet50
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes, mode = "train", ckpt = None):
        super().__init__()
        """
        mode can be train, pretrained, transfer
        """
        model = resnet50()
        if mode in ["pretrained", "transfer"]:
            print(f"loading ResNet50 from {ckpt}")
            ckpt = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(ckpt)
            if mode == "transfer":
                for p in model.parameters():
                    p.requires_grad = False
                pass
        model.fc = nn.Linear(2048, num_classes, bias=True)
        for p in model.fc.parameters():
            p.requires_grad = True
        print(f"ResNet50 parameters: {sum([ p.numel() for p in model.parameters()])}")
        self.model = model


    def forward(self, x):
        """
        x: [B,Ch,224,224]
        output: [B, C]
        """
        return self.model(x)
    
class MobileNet(nn.Module):
    def __init__(self, num_classes, mode = "train", ckpt = None):
        super().__init__()
        """
        mode can be train, pretrained, transfer
        """
        model = mobilenet_v2()
        if mode in ["pretrained", "transfer"]:
            print(f"loading mobilenet from {ckpt}")
            ckpt = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(ckpt)
            if mode == "transfer":
                for p in model.parameters():
                    p.requires_grad = False
                pass

        model.classifier[-1] = nn.Linear(1280, num_classes, bias=True)
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
        print(f"MobileNet parameters: {sum([ p.numel() for p in model.parameters()])}")
        self.model = model


    def forward(self, x):
        """
        x: [B,Ch,224,224]
        output: [B, C]
        """
        return self.model(x)