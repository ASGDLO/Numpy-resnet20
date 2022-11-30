import numpy as np


class AveragePooling:
    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        out_tensor = in_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], -1).mean(axis=-1)
        return out_tensor.reshape(in_tensor.shape[0], in_tensor.shape[1], 1, 1)

    def backward(self, out_diff_tensor, lr):
        batch_num = self.shape[0]
        in_channels = self.shape[1]
        in_h = self.shape[2]
        in_w = self.shape[3]
        assert out_diff_tensor.shape == (batch_num, in_channels, 1, 1)

        in_diff_tensor = np.zeros(list(self.shape))
        in_diff_tensor += out_diff_tensor / (in_h * in_w)

        self.in_diff_tensor = in_diff_tensor