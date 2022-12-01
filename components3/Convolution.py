import numpy as np


class Convolution:
    def __init__(self, in_channels, out_channels, kernel_w, kernel_h,
                 stride=1, pad=0, shift=True, same=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.stride = stride
        self.pad = pad
        self.same = same
        self.shift = shift
        self.kernel = self.init_kernel()
        self.bias = self.init_bias(shift)

    def init_kernel(self):
        return np.random.uniform(
            low=-np.sqrt(6.0 / (self.out_channels + self.in_channels * self.kernel_h * self.kernel_w)),
            high=np.sqrt(6.0 / (self.in_channels + self.out_channels * self.kernel_h * self.kernel_w)),
            size=(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        )

    def init_bias(self, is_shift):
        return np.zeros([self.out_channels]) if is_shift else None

    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = np.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = in_tensor
        return padded

    @staticmethod
    def convolution(in_tensor, kernel, stride=1):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_channels = kernel.shape[0]
        assert kernel.shape[1] == in_channels
        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]

        out_h = int((in_h - kernel_h + 1) / stride)
        out_w = int((in_w - kernel_w + 1) / stride)

        kernel = kernel.reshape(out_channels, -1)

        extend_in = np.zeros([in_channels*kernel_h*kernel_w, batch_num*out_h*out_w])
        for i in range(out_h):
            for j in range(out_w):
                part_in = in_tensor[:, :, i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w].reshape(batch_num, -1)
                extend_in[:, (i*out_w+j)*batch_num:(i*out_w+j+1)*batch_num] = part_in.T

        out_tensor = np.dot(kernel, extend_in)
        out_tensor = out_tensor.reshape(out_channels, out_h*out_w, batch_num)
        out_tensor = out_tensor.transpose(2, 0, 1).reshape(batch_num, out_channels, out_h, out_w)

        return out_tensor

    def forward(self, in_tensor):
        if self.same:
            in_tensor = Convolution.pad(in_tensor, int((self.kernel_h-1)/2), int((self.kernel_w-1)/2))

        self.in_tensor = in_tensor.copy()
        self.out_tensor = Convolution.convolution(in_tensor, self.kernel, self.stride)

        if self.shift:
            self.out_tensor += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape

        if self.shift:
            bias_diff = np.sum(out_diff_tensor, axis=(0, 2, 3)).reshape(self.bias.shape)
            self.bias -= lr * bias_diff

        batch_num = out_diff_tensor.shape[0]
        out_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        extend_out = np.zeros([batch_num, out_channels, out_h, out_w, self.stride * self.stride])
        extend_out[:, :, :, :, 0] = out_diff_tensor
        extend_out = extend_out.reshape(batch_num, out_channels, out_h, out_w, self.stride, self.stride)
        extend_out = extend_out.transpose(0, 1, 2, 4, 3, 5).reshape(batch_num, out_channels, out_h * self.stride,
                                                                    out_w * self.stride)

        kernel_diff = Convolution.convolution(self.in_tensor.transpose(1, 0, 2, 3), extend_out.transpose(1, 0, 2, 3))
        kernel_diff = kernel_diff.transpose(1, 0, 2, 3)

        padded = Convolution.pad(extend_out, self.kernel_h - 1, self.kernel_w - 1)
        kernel_trans = self.kernel.reshape(self.out_channels, self.in_channels, self.kernel_h * self.kernel_w)
        kernel_trans = kernel_trans[:, :, ::-1].reshape(self.kernel.shape)
        self.in_diff_tensor = Convolution.convolution(padded, kernel_trans.transpose(1, 0, 2, 3))
        assert self.in_diff_tensor.shape == self.in_tensor.shape

        if self.same:
            pad_h = int((self.kernel_h - 1) / 2)
            pad_w = int((self.kernel_w - 1) / 2)
            if pad_h == 0 and pad_w != 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, :, pad_w:-pad_w]
            elif pad_h != 0 and pad_w == 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, :]
            elif pad_h != 0 and pad_w != 0:
                self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]

        self.kernel -= lr * kernel_diff