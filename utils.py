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
    
