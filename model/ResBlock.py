from components.BatchNorm import BatchNorm
from components.Convolution import Convolution
from components.Relu import Relu


class ResBlock:
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        self.path1 = [
            Convolution(in_channels, out_channels, 3, 3, stride=stride, shift=False),
            BatchNorm(out_channels),
            Relu(),
            Convolution(out_channels, out_channels, 3, 3, shift=False),
            BatchNorm(out_channels)
        ]
        self.path2 = shortcut
        self.relu = Relu()

    def train(self):
        self.path1[1].train()
        self.path1[4].train()
        if self.path2 is not None:
            self.path2[1].train()

    def eval(self):
        self.path1[1].eval()
        self.path1[4].eval()
        if self.path2 is not None:
            self.path2[1].eval()

    def forward(self, in_tensor):
        x1 = in_tensor.copy()
        x2 = in_tensor.copy()

        for l in self.path1:
            x1 = l.forward(x1)
        if self.path2 is not None:
            for l in self.path2:
                x2 = l.forward(x2)
        self.out_tensor = self.relu.forward(x1+x2)

        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert self.out_tensor.shape == out_diff_tensor.shape

        self.relu.backward(out_diff_tensor, lr)
        x1 = self.relu.in_diff_tensor
        x2 = x1.copy()

        for l in range(1, len(self.path1) + 1):
            self.path1[-l].backward(x1, lr)
            x1 = self.path1[-l].in_diff_tensor

        if self.path2 is not None:
            for l in range(1, len(self.path2) + 1):
                self.path2[-l].backward(x2, lr)
                x2 = self.path2[-l].in_diff_tensor

        self.in_diff_tensor = x1 + x2

    def save(self, path, conv_num, bn_num):
        conv_num = self.path1[0].save(path, conv_num)
        bn_num = self.path1[1].save(path, bn_num)
        conv_num = self.path1[3].save(path, conv_num)
        bn_num = self.path1[4].save(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].save(path, conv_num)
            bn_num = self.path2[1].save(path, bn_num)

        return conv_num, bn_num

    def load(self, path, conv_num, bn_num):
        conv_num = self.path1[0].load(path, conv_num)
        bn_num = self.path1[1].load(path, bn_num)
        conv_num = self.path1[3].load(path, conv_num)
        bn_num = self.path1[4].load(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].load(path, conv_num)
            bn_num = self.path2[1].load(path, bn_num)

        return conv_num, bn_num
