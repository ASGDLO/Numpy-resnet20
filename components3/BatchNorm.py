import os
import numpy as np


class BatchNorm:
    def __init__(self, neural_num, moving_rate=0.1):
        self.gamma = np.random.uniform(low=0, high=1, size=neural_num)
        self.bias = np.zeros([neural_num])
        self.moving_avg = np.zeros([neural_num])
        self.moving_var = np.ones([neural_num])
        self.neural_num = neural_num
        self.moving_rate = moving_rate
        self.is_train = True
        self.epsilon = 1e-5

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def forward(self, in_tensor):
        assert in_tensor.shape[1] == self.neural_num

        self.in_tensor = in_tensor.copy()

        if self.is_train:
            mean = in_tensor.mean(axis=(0, 2, 3))
            var = in_tensor.var(axis=(0, 2, 3))
            self.moving_avg = mean * self.moving_rate + (1 - self.moving_rate) * self.moving_avg
            self.moving_var = var * self.moving_rate + (1 - self.moving_rate) * self.moving_var
            self.var = var
            self.mean = mean
        else:
            mean = self.moving_avg
            var = self.moving_var

        self.normalized = (in_tensor - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.epsilon)
        out_tensor = self.gamma.reshape(1, -1, 1, 1) * self.normalized + self.bias.reshape(1, -1, 1, 1)

        return out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.in_tensor.shape
        assert self.is_train

        m = self.in_tensor.shape[0] * self.in_tensor.shape[2] * self.in_tensor.shape[3]

        normalized_diff = self.gamma.reshape(1, -1, 1, 1) * out_diff_tensor
        var_diff = -0.5 * np.sum(normalized_diff * self.normalized, axis=(0, 2, 3)) / (self.var + self.epsilon)
        mean_diff = -1.0 * np.sum(normalized_diff, axis=(0, 2, 3)) / np.sqrt(self.var + self.epsilon)
        in_diff_tensor1 = normalized_diff / np.sqrt(self.var.reshape(1, -1, 1, 1) + self.epsilon)
        in_diff_tensor2 = var_diff.reshape(1, -1, 1, 1) * (self.in_tensor - self.mean.reshape(1, -1, 1, 1)) * 2 / m
        in_diff_tensor3 = mean_diff.reshape(1, -1, 1, 1) / m
        self.in_diff_tensor = in_diff_tensor1 + in_diff_tensor2 + in_diff_tensor3

        gamma_diff = np.sum(self.normalized * out_diff_tensor, axis=(0, 2, 3))
        self.gamma -= lr * gamma_diff

        bias_diff = np.sum(out_diff_tensor, axis=(0, 2, 3))
        self.bias -= lr * bias_diff

    def save(self, path, bn_num):
        if not os.path.exists(path):
            os.mkdir(path)

        np.save(os.path.join(path, "bn{}_weight.npy".format(bn_num)), self.gamma)
        np.save(os.path.join(path, "bn{}_bias.npy".format(bn_num)), self.bias)
        np.save(os.path.join(path, "bn{}_mean.npy".format(bn_num)), self.moving_avg)
        np.save(os.path.join(path, "bn{}_var.npy".format(bn_num)), self.moving_var)

        return bn_num + 1

    def load(self, path, bn_num):
        assert os.path.exists(path)

        self.gamma = np.load(os.path.join(path, "bn{}_weight.npy".format(bn_num)))
        self.bias = np.load(os.path.join(path, "bn{}_bias.npy".format(bn_num)))
        self.moving_avg = np.load(os.path.join(path, "bn{}_mean.npy".format(bn_num)))
        self.moving_var = np.load(os.path.join(path, "bn{}_var.npy".format(bn_num)))

        return bn_num + 1
