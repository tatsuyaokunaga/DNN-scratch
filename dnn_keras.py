#### ニューラルネットワークの層の構築 ####
batch_size=64
epochs= 20

#### 学習プロセスの設定 ####
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
model.compile(optimizer='adagrad',
                           loss = 'categorical_crossentropy',
                           metrics=['accuracy'])
