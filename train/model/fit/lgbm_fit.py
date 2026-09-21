from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from ..data_loader import data_load
from ..eval.eval_metric import print_metrics

def train_model():
    df = data_load()

    X = df.drop(columns=['기계고장'])
    y = df['기계고장']

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=1)

    model = LGBMClassifier(n_estimators=200,learning_rate=0.2,max_depth=1,random_state=42,verbosity=-1)

    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    print("---------------------------------")
    print("---------------------------------")
    eval = print_metrics(y_test, pred_test, "LightGBM 결과")
    
    return model, eval