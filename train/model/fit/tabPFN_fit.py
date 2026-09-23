from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from ..data_loader import data_load
from ..eval.eval_metric import print_metrics
from dotenv import load_dotenv

def train_model():
    load_dotenv()
    df = data_load()

    X = df.drop(columns=['기계고장'])
    y = df['기계고장']

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=1)

    model = TabPFNClassifier(n_estimators=1, random_state=42, device='cuda', n_preprocessing_jobs=4, ignore_pretraining_limits=True,)

    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    print("---------------------------------")
    print("---------------------------------")
    eval = print_metrics(y_test, pred_test, "TabPFN 결과")

    return model, eval