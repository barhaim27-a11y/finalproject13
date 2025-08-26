TRAIN_DATA_PATH = "data/parkinsons.csv"
MODEL_PATH = "models/parkinsons_production.joblib"
TEMP_MODEL_PATH = "models/parkinsons_candidate.joblib"
BEST_META_PATH = "assets/best_model.json"
RUNS_CSV = "assets/runs.csv"

TARGET = "status"
ALL_COLUMNS = [
    "name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)",
    "MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3",
    "Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1",
    "spread2","D2","PPE"
]
DROP_COLUMNS = ["name", TARGET]
FEATURES = [c for c in ALL_COLUMNS if c not in DROP_COLUMNS]

CV_FOLDS = 5
RANDOM_STATE = 42
SCORING = "roc_auc"

DEFAULT_MODEL = "XGBoost"

MODEL_LIST = [
    "LogisticRegression","RandomForest","ExtraTrees","AdaBoost","GradientBoosting",
    "SVC","LinearSVC","KNN","GaussianNB","LDA","QDA",
    "XGBoost","MLP","KerasNN"
]

DEFAULT_PARAMS = {
    "LogisticRegression": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "solver": "lbfgs"},
    "RandomForest": {"n_estimators": 400, "max_depth": None, "class_weight": "balanced_subsample"},
    "ExtraTrees": {"n_estimators": 400, "max_depth": None, "bootstrap": False},
    "AdaBoost": {"n_estimators": 300, "learning_rate": 0.8},
    "GradientBoosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
    "SVC": {"C": 2.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "probability": True},
    "LinearSVC": {"C": 1.0},
    "KNN": {"n_neighbors": 15, "weights": "distance"},
    "GaussianNB": {},
    "LDA": {},
    "QDA": {},
    "XGBoost": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    "MLP": {"hidden_layer_sizes": (128, 64), "activation": "relu", "learning_rate_init": 1e-3, "max_iter": 600, "early_stopping": True},
    "KerasNN": {"epochs": 60, "batch_size": 32, "hidden1": 128, "hidden2": 64, "dropout": 0.1, "lr": 1e-3}
}

PARAM_GRIDS = {
    "LogisticRegression": {"classifier__C": [0.3, 1.0, 3.0], "classifier__solver": ["lbfgs"]},
    "RandomForest": {"classifier__n_estimators": [300, 500], "classifier__max_depth": [None, 10, 20], "classifier__min_samples_leaf": [1, 2]},
    "ExtraTrees": {"classifier__n_estimators": [300, 500], "classifier__max_depth": [None, 10, 20]},
    "AdaBoost": {"classifier__n_estimators": [200, 400], "classifier__learning_rate": [0.3, 0.8, 1.0]},
    "GradientBoosting": {"classifier__n_estimators": [200, 400], "classifier__learning_rate": [0.03, 0.05, 0.1], "classifier__max_depth": [2, 3, 4]},
    "SVC": {"classifier__C": [0.5, 1.0, 2.0, 4.0], "classifier__gamma": ["scale", 0.01, 0.1], "classifier__kernel": ["rbf"]},
    "LinearSVC": {"classifier__C": [0.5, 1.0, 2.0]},
    "KNN": {"classifier__n_neighbors": [5, 11, 21], "classifier__weights": ["uniform", "distance"]},
    "GaussianNB": {},
    "LDA": {},
    "QDA": {},
    "XGBoost": {"classifier__n_estimators": [300, 500, 700], "classifier__max_depth": [3, 4, 5], "classifier__learning_rate": [0.03, 0.05, 0.1], "classifier__subsample": [0.8, 0.9, 1.0], "classifier__colsample_bytree": [0.8, 1.0]},
    "MLP": {"classifier__hidden_layer_sizes": [(128, 64), (64, 32)], "classifier__alpha": [1e-5, 1e-4], "classifier__learning_rate_init": [1e-3, 5e-4], "classifier__max_iter": [600]},
    "KerasNN": {"classifier__epochs": [40, 80], "classifier__batch_size": [16, 32], "classifier__model__hidden1": [128, 64], "classifier__model__hidden2": [64, 32], "classifier__model__dropout": [0.1, 0.2], "classifier__model__lr": [1e-3, 5e-4]}
}
