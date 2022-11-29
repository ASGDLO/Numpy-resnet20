from PIL import Image
from torchvision import transforms
import random
import numpy as np
import os


class DataLoader:
    def __init__(self, path, filename, batch_size, image_w, image_h):
        self.path = path
        with open(os.path.join(path, filename)) as file:
            self.datalist = file.readlines()
        random.shuffle(self.datalist)
        self.batch_size = batch_size
        self.len = len(self.datalist)
        self.index = 0
        self.image_w = image_w
        self.image_h = image_h
        self.size = (image_w, image_h)

    def reset(self):
        self.index = 0
        random.shuffle(self.datalist)

    # def get_trans_img(self, path):
    #     img = cv2.imread(path)
    #     img = img[:, :, ::-1].astype(np.float32).transpose(2,0,1)
    #     mean = np.mean(img, axis=(1, 2)).reshape(-1, 1, 1)
    #     std = np.std(img, axis=(1, 2)).reshape(-1, 1, 1)
    #     img = (img - mean) / std
    #     return img

    def get_trans_img(self, image):
        mean = [0.80048384, 0.44734452, 0.50106468]
        std = [0.22327253, 0.29523788, 0.24583565]
        transform = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
        path = os.path.join(self.path, image)
        return transform(Image.open(path).convert('RGB'))

    def get_next_batch(self):
        if self.index + self.batch_size >= self.len:
            self.reset()
        images = np.zeros([self.batch_size, 3, self.image_w, self.image_h], dtype=np.float32)
        labels = np.zeros([self.batch_size], dtype=np.int32)
        for i in range(self.batch_size):
            data = self.datalist[i + self.index].strip()
            images[i] = self.get_trans_img(data)
            labels[i] = int(data.split('/')[0])
        self.index += self.batch_size
        return images, labels
