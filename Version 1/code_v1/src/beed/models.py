"""Classical models and hyper-parameter grids."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from . import config as C


def get_models():
    """name -> (estimator, param_grid). Grids are deliberately compact (nested CV cost)."""
    return {
        "Majority": (DummyClassifier(strategy="prior"), {}),
        "LR": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
               {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
        "kNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                {"kneighborsclassifier__n_neighbors": [1, 5, 15, 31],
                 "kneighborsclassifier__weights": ["uniform", "distance"]}),
        "SVM-RBF": (make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True,
                                                        random_state=C.SEED)),
                    {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01]}),
        "RF": (RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=C.SEED),
               {"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 5]}),
        "XGBoost": (XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                                  tree_method="hist", n_jobs=1, random_state=C.SEED,
                                  eval_metric="mlogloss", verbosity=0),
                    {"max_depth": [3, 6], "learning_rate": [0.05, 0.2]}),
        "LightGBM": (LGBMClassifier(n_estimators=300, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, n_jobs=1, random_state=C.SEED,
                                    verbose=-1),
                     {"num_leaves": [15, 63], "learning_rate": [0.05, 0.2]}),
    }


CLASSICAL = ["Majority", "LR", "kNN", "SVM-RBF", "RF", "XGBoost", "LightGBM"]
