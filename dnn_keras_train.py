import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.cross_validation import train_test_split

#60,000枚の28x28，10個の数字の白黒画像と10,000枚のテスト用画像データセット．
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# ラベルデータをカテゴリの1-hotベクトルにエンコードする
n_classes = 10
one_hot_train=keras.utils.to_categorical(t_train , num_classes)
one_hot_test=keras.utils.to_categorical(t_test , num_classes)


history = model.fit(x_train, one_hot_train, # トレーニングデータ
                    batch_size=batch_size,  # バッチサイズ
                    epochs=epochs # 総エポック数
                    )  
