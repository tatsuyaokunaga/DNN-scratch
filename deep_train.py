import numpy as np
from sklearn.model_selection import train_test_split
import deep_network.FiveLayerNet
from utils import get_data　, cross_entropy_error, softmax

X,y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

train_loss_list = []
train_acc_list= []
test_acc_list = []

#hyper-parameter
iter_num = 100000
#バッチ数
batch_size = 16
# 学習率
lr = 0.001
#L2ノルム係数
lam = 0.1

train_size = X_train.shape[0]

iter_per_epoch = max(train_size/batch_size,1)
#input_size, hidden_size, output_sizeはデータセットに合わせタプルで入力
network = FourLayerNet(input_size, hidden_size, output_size)

for i in range(iter_num):
    #ミニバッチ学習、ランダムにミニバッチを取得
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]    
    # 勾配の計算
    grads = network.gradient(x_batch,y_batch)
    
    for key in ('W1','b1','W2','b2','W3','b3','W4','b4'):
       network.params[key] -= lr*grads[key]
    
    loss = network.loss(x_batch,y_batch,lam)
    train_loss_list.append(loss)
    #accuracy = network.accuracy(x_batch,y_batch)
    
    #1エポックごとに精度を計算
    if  i % iter_per_epoch ==0:
        train_acc = network.accuracy(X_train,y_train)
        test_acc = network.accuracy(X_test,y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc, test_acc | " + str(train_acc) + " " + str(test_acc))
        
# accuracy曲線をプロット
def accuracy_plot(train_acc, test_acc):
    plt.figure(figsize=(12,8))
    plt.title("Accuracys vs Iterations")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(test_acc, label="Validation Accuracy")
    plt.legend()
    plt.show()
