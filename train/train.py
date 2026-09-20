from .model.fit.catboost_fit import train_model as catboost_train
from .model.fit.lgbm_fit import train_model as lgbm_train
from .model.fit.xgboost_fit import train_model as xgboost_train

xgboost_model = xgboost_train()
lgbm_model = lgbm_train()
catboost_model = catboost_train()