import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.cross_validation import train_test_split

train = pd.read_csv('./input/train.csv')
test= pd.read_csv('./input/test.csv')
X = train.drop(["label"], axis=1)
y = train["label"]

(x_train, t_train), (x_test, t_test) = mnist.load_data()

# ラベルデータをカテゴリの1-hotベクトルにエンコードする
n_features = X.shape[1]
n_classes = 10
#訓練データとラベルデータをそれぞれ訓練用、テスト用に分割
x_train, x_test ,y_train, y_test = train_test_split(X, y, test_size=0.2)
one_hot_train=keras.utils.to_categorical(y_train , num_classes=10)
one_hot_test=keras.utils.to_categorical(y_test , num_classes=10)


history = model.fit(x_train, one_hot_train, # トレーニングデータ
                    batch_size=batch_size,  # バッチサイズ
                    epochs=epochs # 総エポック数
                    )  
