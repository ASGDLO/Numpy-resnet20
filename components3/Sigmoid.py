import os
import numpy as np


class Sigmoid:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_param()

    def init_param(self):
        self.kernel = np.random.uniform(
            low=-np.sqrt(6.0 / (self.out_channels + self.in_channels)),
            high=np.sqrt(6.0 / (self.in_channels + self.out_channels)),
            size=(self.out_channels, self.in_channels)
        )
        self.bias = np.zeros([self.out_channels])

    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1).copy()
        assert self.in_tensor.shape[1] == self.kernel.shape[1]
        self.out_tensor = np.dot(self.in_tensor, self.kernel.T) + self.bias.T
        self.out_tensor = 1.0 / (1.0 + np.exp(-self.out_tensor))
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        nonlinear_diff = self.out_tensor * (1 - self.out_tensor) * out_diff_tensor
        kernel_diff = np.dot(nonlinear_diff.T, self.in_tensor).squeeze()
        bias_diff = np.sum(nonlinear_diff, axis=0).reshape(self.bias.shape)
        self.in_diff_tensor = np.dot(nonlinear_diff, self.kernel).reshape(self.shape)
        self.kernel -= lr * kernel_diff
        self.bias -= lr * bias_diff

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

        np.save(os.path.join(path, "fc_weight.npy"), self.kernel)
        np.save(os.path.join(path, "fc_bias.npy"), self.bias)

    def load(self, path):
        assert os.path.exists(path)

        self.kernel = np.load(os.path.join(path, "fc_weight.npy"))
        self.bias = np.load(os.path.join(path, "fc_bias.npy"))
