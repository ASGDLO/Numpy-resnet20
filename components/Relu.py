class Relu:
    def forward(self, in_tensor):
        self.in_tensor = in_tensor.copy()
        self.out_tensor = in_tensor.copy()
        self.out_tensor[self.in_tensor < 0.0] = 0.0
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert self.out_tensor.shape == out_diff_tensor.shape
        self.in_diff_tensor = out_diff_tensor.copy()
        self.in_diff_tensor[self.in_tensor < 0.0] = 0.0