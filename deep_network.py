class FourLayerNet:
    
    def __init__(self,input_size, hidden_size,output_size,lam=0.1,lr=0.01):
        #重みの初期化
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.random.randn(hidden_size,hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)
        self.lam = lam
        self.lr = lr
        
    def predict(self,x):
        W1,W2,W3,W4 = self.params['W1'],self.params['W2'] ,self.params['W3'], self.params['W4']
        b1,b2,b3,b4 = self.params['b1'],self.params['b2'],self.params['b3'],self.params['b4']
        
        z1 = np.dot(x,W1)+b1
        a1=np.tanh(z1)
        z2=np.dot(a1,W2)+b2
        a2=np.tanh(z2)
        z3=np.dot(a2,W3)+b3
        a3=np.tanh(z3)
        z4=np.dot(a3,W4)+b4
        
        y = softmax(z4)
               
        return y
    
    def loss(self,x,t,lam):
        W1,W2,W3,W4 = self.params['W1'],self.params['W2'], self.params['W3'], self.params['W4']
        b1,b2,b3,b4 = self.params['b1'],self.params['b2'],self.params['b3'],self.params['b4']
        y = self.predict(x)
        
        return cross_entropy_error(y,t,W1,W2,W3,W4,lam)
    
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        
        return accuracy
    
    def gradient(self,x,t):
        self.lam
        W1,W2,W3,W4 = self.params['W1'],self.params['W2'] ,self.params['W3'],self.params['W4']
        b1,b2,b3,b4 = self.params['b1'],self.params['b2'],self.params['b3'],self.params['b4']
        grads={}
        
        batch_num = x.shape[0]
        #forward
        
        z1 = np.dot(x,W1)+b1
        a1=np.tanh(z1)
        z2=np.dot(a1,W2)+b2
        a2=np.tanh(z2)
        z3=np.dot(a2,W3)+b3
        a3=np.tanh(z3)
        z4=np.dot(a3,W4)+b4
        
        y = softmax(z4)
        
        #backward
        dy = (y-t)/batch_num
    
        grads['W4'] = np.dot(a3.T, dy) + self.lam*W4
        grads['b4']  = dy.sum(axis=0)
 
        da3 = (1 - np.tanh(z3)**2)*(dy.dot(W4.T))
        grads['W3'] = np.dot(a2.T,da3) + self.lam*W3
        grads['b3']  = da3.sum(axis=0)
        
        da2 = (1 - np.tanh(z2)**2)*(da3.dot(W3.T))
        grads['W2'] = np.dot(a1.T,da2) + self.lam*W2
        grads['b2']  = da2.sum(axis=0)
        
        da1 = (1 - np.tanh(z1)**2)*(da2.dot(W2.T))
        grads['W1'] = np.dot(x.T,da1) + self.lam*W1
        grads['b1']  = da1.sum(axis=0)
        
        return grads
