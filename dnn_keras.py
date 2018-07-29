import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,BatchNormalization
from keras.optimizers import Adagrad,SGD


#### ニューラルネットワークの層の構築 ####
# 最初のlayerでは，想定する入力データshapeを784次元(mnistデータセット用)に指定
# Sequenceモデル
model= Sequential()
model.add(Dense(128,input_shape=(784,)))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('tanh'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(Dense(256))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('softmax'))

# モデルの学習処理の設定
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
batch_size=64
epochs= 20
model.compile(optimizer='adagrad',
                           loss = 'categorical_crossentropy',
                           metrics=['accuracy'])



### Functional APIで実装
from keras.layers import Input, Dense,BatchNormalization,Dropout
from keras.models import Model

inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(128, activation='tanh')(inputs)
x = Dropout(.25)(x)
x = Dense(256, activation='tanh')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, activation='tanh')(x)
x = Dropout(.25)(x)
#x = Dense(128, activation='tanh', kernel_initializer=he_normal())(x)
#x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

batch_size=64
epochs= 20
# モデルの学習処理の設定
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
