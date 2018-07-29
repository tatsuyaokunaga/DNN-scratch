import pandas as pd

def get_data():
    # ローカルに保存済みのデータセット(このフォルダと同じ階層のinputフォルダ内に「train.csv」「test.csv」を想定）を読み込む
    train = pd.read_csv('../input/train.csv')
    test= pd.read_csv('../input/test.csv')
    X = train.drop(["label"],axis=1)
    X =X.values
    y = np.array(train["label"])

    n_labels = len(np.unique(y))  # 分類クラスの数
    y = np.eye(n_labels)[y]
    
    return X,y
 
# コスト関数に正則化のためL2ノルムを加算
def cross_entropy_error(y,t, W1, W2,W3,W4, lam):
    batch_size= y.shape[0]
    return -(np.sum(t*np.log(y + 1e-7))+(lam/2)*((W1**2).sum()+(W2**2).sum() +
                                                  (W3**2).sum()+(W4**2).sum()))/batch_size 

# ソフトマックス関数
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
