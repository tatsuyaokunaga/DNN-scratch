


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
        t = Variable(y_train[perm[i:i+batch_size]])
        y_pred = model.forward(x)
        model.cleargrads()
        loss = F.softmax_cross_entropy(y_pred, t)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data*batch_size

    print("epoch: {}, mean loss: {}".format(epoch, sum_loss/len(x_train)))
