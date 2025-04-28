'''
Description: 
    1）创建dataloader
    2) 创建神经网络
    3) 训练模型
    4) 展示效果
Author: KuhnLiu
Date: 2024-07-11 10:04:42
LastEditTime: 2024-07-14 11:42:31
LastEditors: KuhnLiu
'''
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
        #print(images.shape, labels.shape)
        images = images.astype("float32") / 255
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
        #print(np.eye(10))
        labels = np.eye(10)[labels]
        return images, labels
        
if __name__ == "__main__":
    # 通过dataloader读取数据
    dataloader = Dataloader()
    images, labels = dataloader.get_data()
    print(images.shape, labels.shape)
    
    # 创建模型
    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    b_i_h = np.zeros((20, 1))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_h_o = np.zeros((10, 1))
    # 训练模型
    # - 设置超参数
    lern_rate = 0.01
    nr_correct = 0
    epochs = 5
    
    # - 向前传播
    for epoch in range(epochs):
        for img, l in zip(images, labels):
            # print('img.shape',img.shape)
            img.shape += (1,)
            # print('img.shape',img.shape)
            # print('l.shape',l.shape)
            l.shape += (1,)
            # print('l.shape',l.shape)
            h_pre = b_i_h + w_i_h @ img
            # print('h_pre.shape',h_pre.shape)
            h = 1 / (1 + np.exp(-h_pre))
            o_pre = b_h_o + w_h_o @ h
            # print('o_pre.shape',o_pre.shape)
            o = 1 / (1 + np.exp(-o_pre))
            # print('o.shape',o.shape)
    # - 损失函数计算
            # print('o',o)
            # print('l',l)
            # print('o - l',o - l)
            # print('(o - l) ** 2',(o - l) ** 2)
            e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
            # print('e',e)
            nr_correct += int(np.argmax(o) == np.argmax(l))

    # - 反向传播
            # 反向传播中，神经层的调整量是其到下一层神经的权重。神经层到下一层的权重的调整比例是神经层

            # 输出层调整量是损失函数的导数
            delta_o = 0.2 * (o - l)
            # print("delta_o.shape",delta_o.shape)
            # 隐藏层到输出层过程中权重激活函数的导数
            delta_z = (o * (1 - o))
            # 隐藏层到输出层过程中的权重要调整的比例=隐藏层的数值
            delta_w_h_o = np.transpose(h)
            # 隐藏层到输出层过程中的权重要调整的值就是 输出层调整量*权重层要调整的比例*激活函数的导数
            w_h_o += -lern_rate * delta_o @ delta_w_h_o * delta_z
            # 隐藏层到输出层过程中的权重的偏移量就是输出层调整量
            b_h_o += -lern_rate * delta_o

            # 隐藏层调量是隐藏层到输出层的权重*输出层的调整量
            delta_h = np.transpose(w_h_o) @ delta_o
            # print("delta_h.shape",delta_h.shape)
            # 输入层到隐藏层的权重的激活函数的导数
            delta_z_2 = (h * (1 - h))
            # 输入层到隐藏层的权重要调整的比例=输入层的数值
            delta_w_i = np.transpose(img)
            # print('h.shape',h.shape)
            # print('delta_z_2.shape',delta_z_2.shape)
            # print('delta_w_i.shape',delta_w_i.shape)
            # 输入层到隐藏层的权重要调整的值就是 隐藏层调整量*输入层到隐藏层的权重的激活函数的导数*输入层到隐藏层的权重要调整的比例
            w_i_h += -lern_rate * delta_h @ delta_w_i * delta_z_2
            b_i_h += -lern_rate * delta_h

    # - 输出精准度
            
            # break
        print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%") 
        nr_correct = 0
        # break
        
    # - 展示效果
    while True:
        index = int(input("输入编号进行预测 (0 - 59999): "))
        img = images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        img.shape += (1,)
        # 前向传播
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        plt.title(f"This figure is predicted to be: {o.argmax()} :")
        plt.show()