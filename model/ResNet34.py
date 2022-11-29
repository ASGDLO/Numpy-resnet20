from components.Convolution import Convolution
from components.BatchNorm import BatchNorm
from components.MaxPooling import MaxPooling
from components.Relu import Relu
from components.Sigmoid import Sigmoid
from components.AveragePooling import AveragePooling
from model.ResBlock import ResBlock


class ResNet34:
    def __init__(self, num_classes):
        self.pre = [
            Convolution(3, 64, 7, 7, stride=2, shift=False),
            BatchNorm(64),
            Relu(),
            MaxPooling(3, 3, 2, same=True)
        ]
        self.layer1 = self.stack_ResBlock(64, 64, 3, 1)
        self.layer2 = self.stack_ResBlock(64, 128, 4, 2)
        self.layer3 = self.stack_ResBlock(128, 256, 6, 2)
        self.layer4 = self.stack_ResBlock(256, 512, 3, 2)
        self.avg = AveragePooling()
        self.fc = Sigmoid(512, num_classes)

    def train(self):
        self.pre[1].train()
        for l in self.layer1:
            l.train()
        for l in self.layer2:
            l.train()
        for l in self.layer3:
            l.train()
        for l in self.layer4:
            l.train()

    def eval(self):
        self.pre[1].eval()
        for l in self.layer1:
            l.eval()
        for l in self.layer2:
            l.eval()
        for l in self.layer3:
            l.eval()
        for l in self.layer4:
            l.eval()

    def stack_ResBlock(self, in_channels, out_channels, block_num, stride):
        shortcut = [
            Convolution(in_channels, out_channels, 1, 1, stride=stride, shift=False),
            BatchNorm(out_channels)
        ]
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride, shortcut=shortcut))

        for _ in range(block_num - 1):
            layers.append(ResBlock(out_channels, out_channels))

        return layers

    def forward(self, in_tensor):
        x = in_tensor
        for l in self.pre:
            x = l.forward(x)
        for l in self.layer1:
            x = l.forward(x)
        for l in self.layer2:
            x = l.forward(x)
        for l in self.layer3:
            x = l.forward(x)
        for l in self.layer4:
            x = l.forward(x)
        x = self.avg.forward(x)
        out_tensor = self.fc.forward(x)

        return out_tensor

    def backward(self, out_diff_tensor, lr):
        x = out_diff_tensor
        self.fc.backward(x, lr)
        x = self.fc.in_diff_tensor
        self.avg.backward(x, lr)
        x = self.avg.in_diff_tensor

        for l in range(1, len(self.layer4) + 1):
            self.layer4[-l].backward(x, lr)
            x = self.layer4[-l].in_diff_tensor
        for l in range(1, len(self.layer3) + 1):
            self.layer3[-l].backward(x, lr)
            x = self.layer3[-l].in_diff_tensor
        for l in range(1, len(self.layer2) + 1):
            self.layer2[-l].backward(x, lr)
            x = self.layer2[-l].in_diff_tensor
        for l in range(1, len(self.layer1) + 1):
            self.layer1[-l].backward(x, lr)
            x = self.layer1[-l].in_diff_tensor
        for l in range(1, len(self.pre) + 1):
            self.pre[-l].backward(x, lr)
            x = self.pre[-l].in_diff_tensor
        self.in_diff_tensor = x

    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], -1)
        return np.argmax(out_tensor, axis=1)

    def save(self, path):
        conv_num = 0
        bn_num = 0

        if os.path.exists(path) == False:
            os.mkdir(path)

        conv_num = self.pre[0].save(path, conv_num)
        bn_num = self.pre[1].save(path, bn_num)

        for l in self.layer1:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.save(path, conv_num, bn_num)

        self.fc.save(path)

    def load(self, path):
        conv_num = 0
        bn_num = 0

        conv_num = self.pre[0].load(path, conv_num)
        bn_num = self.pre[1].load(path, bn_num)

        for l in self.layer1:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.load(path, conv_num, bn_num)

        self.fc.load(path)