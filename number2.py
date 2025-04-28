import numpy as np
import pathlib
import matplotlib.pyplot as plt

class Dataloader():
    """
        数据读取器
    """  
    def get_data(self):
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/number2-mnist.npz") as f:
            images, labels = f["x_train"], f["y_train"]
        images = images.astype("float32") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        labels = np.eye(10)[labels]
        return images, labels
        

if __name__ == "__main__":
    # 通过dataloader读取数据
    dataloader = Dataloader()
    images, labels = dataloader.get_data()
    # 创建模型
    # 本代码重在快速实现神经网络，因此对模型不做封装
    # 输入层到隐藏层1的权重
    w_i_h1 = np.random.uniform(-0.5, 0.5, (20, 784))
    b_i_h1 = np.zeros((20, 1))
    # 隐藏层1到隐藏层2的权重
    w_h1_h2 = np.random.uniform(-0.5, 0.5, (20, 20))
    b_h1_h2 = np.zeros((20, 1))
    # 隐藏层2到输出层的权重
    w_h2_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_h2_o = np.zeros((10, 1))
    # 训练模型
    # - 设置超参数
    learn_rate = 0.01
    nr_correct = 0
    epochs = 5    
    for epoch in range(epochs):
        for img, l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)
            # 前向传播
            # - 输入层 -> 隐藏层1
            h1_pre = b_i_h1 + w_i_h1 @ img
            # 激活函数
            h1 = 1 / (1 + np.exp(-h1_pre))
            # print('h1.shape', h1.shape)
            # - 隐藏层1 -> 隐藏层2
            h2_pre = b_h1_h2 + w_h1_h2 @ h1
            # 激活函数
            h2 = 1 / (1 + np.exp(-h2_pre))
            # print('h2.shape', h2.shape)
            # - 隐藏层2 -> 输出层
            h3_pre = b_h2_o + w_h2_o @ h2
            # 激活函数
            o = 1 / (1 + np.exp(-h3_pre))
            # - 损失函数计算 相当于(o - l) * np.transpose(o - l)
            e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
            nr_correct += int(np.argmax(o) == np.argmax(l))
            # 反向传播
            # - 损失 -> 隐藏层2到输出层
            # 损失函数1 / len(o) * (o - l) ** 2 的导数，len(o)=10
            delta_o = 0.2 * (o - l)
            # 激活函数的导数
            delta_zo = (o * (1 - o))
            delta_w_h2 = np.transpose(h2)

            # print('delta_o.shape', delta_o.shape)
            # print('delta_zo.shape', delta_zo.shape)
            # print('delta_w_h2.shape', delta_w_h2.shape)
            # print('(-learn_rate * delta_o @ delta_w_h2 * delta_zo).shape', (-learn_rate * delta_o @ delta_w_h2 * delta_zo).shape )
            # print('(-learn_rate * delta_o).shape', (-learn_rate * delta_o).shape )
            # print('(delta_w_h2 * delta_zo ).shape', (delta_w_h2 * delta_zo ).shape )

            #           (1)         (10, 1)     (1, 20)      (10, 1)
            w_h2_o += -learn_rate * delta_o @ delta_w_h2 * delta_zo 
            b_h2_o += -learn_rate * delta_o


            # - 隐藏层2 -> 隐藏层1
            delta_h2 = np.transpose(w_h2_o) @ delta_o
            # 激活函数的导数
            delta_z2 = (h2 * (1 - h2))
            delta_w_h1_h2 = np.transpose(h1)

            # print('delta_o.shape', delta_o.shape)
            # print('delta_z2.shape', delta_z2.shape)
            # print('delta_w_h1_h2.shape', delta_w_h1_h2.shape)

            w_h1_h2 += -learn_rate * delta_h2 @ delta_w_h1_h2 * delta_z2
            b_h1_h2 += -learn_rate * delta_h2

            # - 隐藏层1 -> 输入层
            delta_h1 = np.transpose(w_h1_h2) @ delta_h2
            delta_z1 = (h1 * (1 - h1))
            delta_w_i_h1 = np.transpose(img)

            # print('xxx.shape', (-learn_rate * delta_h2 @ delta_o * delta_z2).shape)

            # print('delta_w_h1_h2.shape', delta_w_h1_h2.shape)
            # print('delta_h1.shape', delta_h1.shape)
            # print('delta_z1.shape', delta_z1.shape)
            # print('delta_h2.shape', delta_h2.shape)
            # print('delta_w_i_h1.shape', delta_w_i_h1.shape)
            # print('xxx2.shape', (-learn_rate * delta_h1 @ delta_h2 * delta_z1).shape)

            #           (1)          (20, 20)   (20, 10)  (20, 1)     (1, 784)
            w_i_h1 += -learn_rate * delta_h1 @ delta_w_i_h1 * delta_z1
            b_i_h1 += -learn_rate * delta_h1

        # 输出精准度
        print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0
    # 展示效果
    while True:
        index = int(input("输入编号进行预测 (0 - 59999): "))
        img = images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        img.shape += (1,)
        # 前向传播
        h1_pre = b_i_h1 + w_i_h1 @ img.reshape(784, 1)
        h1 = 1 / (1 + np.exp(-h1_pre))
        h2_pre = b_h1_h2 + w_h1_h2 @ h1
        h2 = 1 / (1 + np.exp(-h2_pre))
        o_pre = b_h2_o + w_h2_o @ h2
        o = 1 / (1 + np.exp(-o_pre))
        plt.title(f"This figure is predicted to be: {o.argmax()} :")
        plt.show()
   


