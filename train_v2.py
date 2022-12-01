from dataloader3.DataLoader import DataLoader
from model3.ResNet34 import ResNet34
from test import test
from trainer.Trainer import Trainer
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    batch_size = 8
    image_h = 64
    image_w = 64
    dataset = DataLoader("train.txt", batch_size, image_h, image_w)

    model = ResNet34(20)

    init_lr = 0.01
    train = Trainer(model, dataset, 20, init_lr)
    loss = []
    accurate = []
    temp = 0
    start_time = time.time()

    model.train()
    plt.figure(figsize=(10, 5))
    plt.ion()
    for i in range(1000):
        temp += train.iterate()
        if i % 10 == 0 and i != 0:
            loss.append(temp / 10)
            print("iteration = {} || loss = {}".format(str(i), str(temp/10)))
            print("--- %s seconds ---" % (time.time() - start_time))   
            temp = 0
            if i % 100 == 0:
                model.eval()
                accurate.append(test(model, "test.txt", image_h, image_w))
                model.save("model2")
                model.train()
        plt.cla()
        plt.subplot(1, 2, 1)
        plt.plot(loss)
        plt.subplot(1, 2, 2)
        plt.plot(accurate)
        plt.pause(0.1)

        if i == 15000:
            train.set_lr(0.001)

    plt.ioff()
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))