import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.cross_validation import train_test_split

#60,000枚の28x28，10個の数字の白黒画像と10,000枚のテスト用画像データセット．
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# ラベルデータをカテゴリの1-hotベクトルにエンコードする
# n_classes = 10
# one_hot_train=keras.utils.to_categorical(t_train , num_classes)
# one_hot_test=keras.utils.to_categorical(t_test , num_classes)

# 各データセットをndarray型に変換
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
t_train = np.array(t_train, dtype=np.int32)
t_test = np.array(t_test, dtype=np.int32)

#学習の最適化手法、エポック数、バッチサイズを設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# エポック数
n_epoch = 10
# バッチサイズ
batch_size = 100

# 学習実行
for epoch in range(n_epoch):
    sum_loss = 0
    perm = np.random.permutation(len(x_train))
    for i in range(0, len(x_train), batch_size):
        x = Variable(x_train[perm[i:i+batch_size]])
        t = Variable(t_train[perm[i:i+batch_size]])
        y_pred = model.forward(x)
        model.cleargrads()
        loss = F.softmax_cross_entropy(y_pred, t)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data*batch_size

    print("epoch: {}, mean loss: {}".format(epoch, sum_loss/len(x_train)))
