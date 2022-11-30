import numpy as np


class MaxPooling:
    def __init__(self, kernel_h, kernel_w, stride, same=False):
        assert stride > 1
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.same = same

    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = np.zeros([batch_num, in_channels, in_h + 2 * pad_h, in_w + 2 * pad_w])
        padded[:, :, pad_h:pad_h + in_h, pad_w:pad_w + in_w] = in_tensor

        return padded

    def forward(self, in_tensor):
        if self.same:
            in_tensor = MaxPooling.pad(in_tensor, int((self.kernel_h - 1) / 2), int((self.kernel_w - 1) / 2))
        self.shape = in_tensor.shape

        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_h = int((in_h - self.kernel_h) / self.stride) + 1
        out_w = int((in_w - self.kernel_w) / self.stride) + 1

        out_tensor = np.zeros([batch_num, in_channels, out_h, out_w])
        self.maxindex = np.zeros([batch_num, in_channels, out_h, out_w], dtype=np.int32)
        for i in range(out_h):
            for j in range(out_w):
                part = in_tensor[:, :, i * self.stride:i * self.stride + self.kernel_h,
                       j * self.stride:j * self.stride + self.kernel_w].reshape(batch_num, in_channels, -1)
                out_tensor[:, :, i, j] = np.max(part, axis=-1)
                self.maxindex[:, :, i, j] = np.argmax(part, axis=-1)
        self.out_tensor = out_tensor
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        in_h = self.shape[2]
        in_w = self.shape[3]

        out_diff_tensor = out_diff_tensor.reshape(batch_num * in_channels, out_h, out_w)
        self.maxindex = self.maxindex.reshape(batch_num * in_channels, out_h, out_w)

        self.in_diff_tensor = np.zeros([batch_num * in_channels, in_h, in_w])
        h_index = (self.maxindex / self.kernel_h).astype(np.int32)
        w_index = self.maxindex - h_index * self.kernel_h
        for i in range(out_h):
            for j in range(out_w):
                self.in_diff_tensor[range(batch_num * in_channels),
                                    i * self.stride + h_index[:, i, j],
                                    j * self.stride + w_index[:, i, j]] += out_diff_tensor[:, i, j]
        self.in_diff_tensor = self.in_diff_tensor.reshape(batch_num, in_channels, in_h, in_w)

        if self.same:
            pad_h = int((self.kernel_h - 1) / 2)
            pad_w = int((self.kernel_w - 1) / 2)
            self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]