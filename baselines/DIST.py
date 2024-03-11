import torch
from torch.nn import functional as F
class Layer(torch.nn.Module):
    def __init__(self,in_c):
        super(Layer,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_c, 64, kernel_size=5, padding=2)
        self.norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        y = F.relu(x + self.conv3(self.norm2(self.conv2(F.relu(x)))))
        return y

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = Layer(1)
        self.layer2 = Layer(64)
        self.layer3 = Layer(64)
        self.layer4 = Layer(64)
        self.layer5 = Layer(64)
        self.layer6 = Layer(64)
        self.conv = torch.nn.Conv2d(64, 4, kernel_size=5, padding=2)

    def forward(self, x):
        c_x = self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        c_y = self.conv(c_x) + x
        return c_y