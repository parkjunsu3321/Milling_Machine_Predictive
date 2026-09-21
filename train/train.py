import joblib
import json
import os

from .model.fit.catboost_fit import train_model as catboost_train
from .model.fit.lgbm_fit import train_model as lgbm_train
from .model.fit.xgboost_fit import train_model as xgboost_train
from .model.fit.tabPFN_fit import train_model as tabPFN_train


models = {}

models["catboost"] = catboost_train()
models["lgbm"] = lgbm_train()
models["xgboost"] = xgboost_train()
models["tabPFN"] = tabPFN_train()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "model", "models")
eval_dir = os.path.join(current_dir, "model", "eval", "data")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

for model_name, model in models.items():
    model_path = os.path.join(model_dir,f"{model_name}_model.pkl")
    eval_path = os.path.join(eval_dir,f"{model_name}_eval.json")
    
    joblib.dump(model, model_path)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(models[model_name][1], f, ensure_ascii=False, indent=4)
        
    print(f"{model_name} 모델 저장 완료: {model_path}")
    print(f"{model_name} 평가 데이터 저장 완료: {eval_path}")