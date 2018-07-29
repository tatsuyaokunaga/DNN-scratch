import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


# DNN クラスを作成
class DNN(Chain):
    def __init__(self):
        super(DNN,self).__init__(
        l1 = L.Linear(784,100),
        l2 = L.Linear(100,100),
        l3 = L.Linear(100,100),
        l4 = L.Linear(100,100),
        l5 = L.Linear(100,10)
        )
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = self.l5(h4)
        return h5

# DNNクラスのインスタンス
model = DNN()
