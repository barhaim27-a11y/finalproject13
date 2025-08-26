from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import os, json, shutil, time, csv
import numpy as np, pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

def _ensure_dirs():
    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path("assets").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

def _generate_synthetic(n: int = 195, seed: int = 29) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    def clip(a,b,c): return np.clip(a,b,c)
    Fo = clip(rng.normal(150,30,n),80,300)
    Fhi = Fo + clip(rng.normal(20,20,n),5,100)
    Flo = Fo - clip(rng.normal(20,20,n),5,100)
    jitterp = clip(rng.normal(0.005,0.003,n),0.0005,0.03)
    jitterabs = clip(rng.normal(0.00005,0.00003,n),0.000005,0.001)
    rap = jitterp * rng.uniform(0.6,1.4,n)
    ppq = jitterp * rng.uniform(0.5,1.5,n)
    ddp = rap*3
    shimmer = clip(rng.normal(0.03,0.015,n),0.005,0.2)
    shimmerdb = clip(rng.normal(0.3,0.2,n),0.02,1.5)
    apq3 = shimmer * rng.uniform(0.6,1.2,n)
    apq5 = shimmer * rng.uniform(0.6,1.2,n)
    apq = shimmer * rng.uniform(0.7,1.3,n)
    dda = apq3*3
    nhr = clip(rng.normal(0.03,0.02,n),0.001,0.3)
    hnr = clip(rng.normal(21,5,n),5,40)
    status = rng.integers(0,2,n)
    rpde = rng.uniform(0.2,0.8,n)
    dfa = rng.uniform(0.5,0.9,n)
    spread1 = rng.normal(-5,2,n)
    spread2 = rng.normal(2.5,1,n)
    d2 = rng.uniform(1.0,3.0,n)
    ppe = rng.uniform(0.1,0.8,n)
    name = [f"synthetic_{i}" for i in range(n)]
    return pd.DataFrame({
        "name": name, "MDVP:Fo(Hz)": Fo, "MDVP:Fhi(Hz)": Fhi, "MDVP:Flo(Hz)": Flo,
        "MDVP:Jitter(%)": jitterp, "MDVP:Jitter(Abs)": jitterabs, "MDVP:RAP": rap, "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp, "MDVP:Shimmer": shimmer, "MDVP:Shimmer(dB)": shimmerdb, "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5, "MDVP:APQ": apq, "Shimmer:DDA": dda, "NHR": nhr, "HNR": hnr, "status": status,
        "RPDE": rpde, "DFA": dfa, "spread1": spread1, "spread2": spread2, "D2": d2, "PPE": ppe
    })

def validate_training_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    required = config.FEATURES + [config.TARGET]
    miss = [c for c in required if c not in df.columns]
    if miss:
        errors.append("Missing columns: " + ", ".join(miss))
        return False, errors
    for col in config.FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column {col} must be numeric")
    if not pd.api.types.is_numeric_dtype(df[config.TARGET]):
        errors.append(f"Target {config.TARGET} must be numeric")
    return len(errors)==0, errors

def load_data(path: str) -> pd.DataFrame:
    _ensure_dirs()
    p = Path(path)
    if not p.exists():
        df = _generate_synthetic()
        df.to_csv(p, index=False)
        return df
    return pd.read_csv(p)

def _preprocessor() -> ColumnTransformer:
    feats = config.FEATURES
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, feats)], remainder="drop")

def _make_classifier(name: str, params: Dict[str, Any]):
    try:
        if name == "LogisticRegression":
            return LogisticRegression(random_state=config.RANDOM_STATE, **params)
        if name == "RandomForest":
            return RandomForestClassifier(random_state=config.RANDOM_STATE, **params)
        if name == "ExtraTrees":
            return ExtraTreesClassifier(random_state=config.RANDOM_STATE, **params)
        if name == "AdaBoost":
            return AdaBoostClassifier(random_state=config.RANDOM_STATE, **params)
        if name == "GradientBoosting":
            return GradientBoostingClassifier(random_state=config.RANDOM_STATE, **params)
        if name == "SVC":
            return SVC(random_state=config.RANDOM_STATE, **params)
        if name == "LinearSVC":
            base = LinearSVC(random_state=config.RANDOM_STATE, **params)
            return CalibratedClassifierCV(base, method="sigmoid")
        if name == "KNN":
            return KNeighborsClassifier(**params)
        if name == "GaussianNB":
            return GaussianNB(**params)
        if name == "LDA":
            return LDA(**params)
        if name == "QDA":
            return QDA(**params)
        if name == "XGBoost":
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                raise RuntimeError(f"XGBoost is not available in this environment: {e}")
            d = dict(eval_metric="logloss", random_state=config.RANDOM_STATE, tree_method="hist")
            d.update(params or {})
            return XGBClassifier(**d)
        if name == "MLP":
            return MLPClassifier(random_state=config.RANDOM_STATE, **params)
        if name == "KerasNN":
            try:
                from scikeras.wrappers import KerasClassifier
                import tensorflow as tf
            except Exception as e:
                raise RuntimeError(f"Keras/TensorFlow not available: {e}")
            def build_keras_model(n_features_in_, hidden1=128, hidden2=64, dropout=0.1, lr=1e-3):
                inputs = tf.keras.Input(shape=(n_features_in_,))
                x = tf.keras.layers.Dense(hidden1, activation="relu")(inputs)
                x = tf.keras.layers.Dropout(dropout)(x)
                x = tf.keras.layers.Dense(hidden2, activation="relu")(x)
                outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
                m = tf.keras.Model(inputs, outputs)
                m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss="binary_crossentropy", metrics=["AUC"])
                return m
            return KerasClassifier(model=build_keras_model, **(params or {}))
    except Exception as e:
        raise e
    raise ValueError(f"Unknown model: {name}")

def create_pipeline(model_name: str, model_params: dict, use_smote: bool=False, calibrate: bool=False) -> Pipeline:
    pre = _preprocessor()
    clf = _make_classifier(model_name, model_params)
    if use_smote:
        pipe = ImbPipeline([("preprocessor", pre), ("smote", SMOTE(random_state=config.RANDOM_STATE)), ("classifier", clf)])
    else:
        pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])
    if calibrate:
        pipe = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    return pipe

def _get_proba(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X); return (d-d.min())/(d.max()-d.min()+1e-8)
    return model.predict(X)

def _save_plots(y_true, y_proba, model_name: str, tag: Optional[str]=None, normalize_cm: bool=True):
    assets = Path("assets"); assets.mkdir(parents=True, exist_ok=True)
    suf = f"_{tag}" if tag else ""
    fpr, tpr, _ = roc_curve(y_true, y_proba); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {model_name}"); plt.legend(); plt.savefig(assets/f"roc{suf}.png", dpi=150, bbox_inches="tight"); plt.close()
    prec, rec, _ = precision_recall_curve(y_true, y_proba); ap = average_precision_score(y_true, y_proba)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {model_name}"); plt.legend(); plt.savefig(assets/f"pr{suf}.png", dpi=150, bbox_inches="tight"); plt.close()
    thr_pred = (y_proba>=0.5).astype(int)
    cm = confusion_matrix(y_true, thr_pred, normalize="true" if normalize_cm else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format=".2f" if normalize_cm else "d"); plt.title("Confusion Matrix"); plt.savefig(assets/f"cm{suf}.png", dpi=150, bbox_inches="tight"); plt.close()
    return {"roc_path": str(assets/f"roc{suf}.png"), "pr_path": str(assets/f"pr{suf}.png"), "cm_path": str(assets/f"cm{suf}.png"),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "prec": prec.tolist(), "rec": rec.tolist()}

def best_threshold(y_true, y_proba, mode="youden"):
    from sklearn.metrics import f1_score
    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    best, best_t = -1, 0.5
    for t in np.linspace(0,1,201):
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best: best, best_t = f1, t
    return float(best_t)

def _log_run(model_name: str, params: Dict[str, Any], metrics: Dict[str, Any], candidate_path: str, duration_s: float):
    Path(config.RUNS_CSV).parent.mkdir(parents=True, exist_ok=True)
    run = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model_name, "duration_s": round(duration_s,3)}
    for k,v in (params or {}).items(): run[f"param_{k}"] = v
    for k,v in metrics.items(): run[f"metric_{k}"] = v
    run["candidate_path"] = candidate_path
    write_header = not Path(config.RUNS_CSV).exists()
    with open(config.RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(run.keys()))
        if write_header: w.writeheader()
        w.writerow(run)

def train_model(data_path: str, model_name: str, model_params: dict, test_size: float=0.2,
                do_cv: bool=True, do_tune: bool=True, artifact_tag: Optional[str]=None,
                use_groups: bool=False, use_smote: bool=False, calibrate: bool=False,
                thr_mode: str="youden") -> Dict[str, Any]:
    _ensure_dirs()
    t0 = time.perf_counter()
    try:
        df = load_data(data_path)
        ok, errs = validate_training_data(df)
        if not ok: return {"ok": False, "errors": errs}

        X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
        if use_groups and "name" in df.columns:
            groups = df["name"]
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_STATE)
            tr_idx, val_idx = next(gss.split(X, y, groups))
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]; y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            cv = GroupKFold(n_splits=config.CV_FOLDS) if do_cv else None
            cv_groups = groups.iloc[tr_idx]
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE) if do_cv else None
            cv_groups = None

        pipe = create_pipeline(model_name, model_params, use_smote=use_smote, calibrate=calibrate)

        cv_means = None
        if cv is not None:
            scoring = ["roc_auc","accuracy","f1","precision","recall"]
            try:
                scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1,
                                        return_train_score=False, groups=cv_groups, error_score="raise")
            except Exception:
                try:
                    scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1,
                                            return_train_score=False, groups=cv_groups, error_score=np.nan)
                except Exception as e2:
                    scores = None
            if scores is not None:
                cv_means = {}
                for m in ["roc_auc","accuracy","f1","precision","recall"]:
                    arr = scores.get(f"test_{m}", None)
                    if arr is None:
                        cv_means[m] = float("nan")
                    else:
                        cv_means[m] = float(np.nanmean(arr))
                if all(np.isnan(list(cv_means.values()))):
                    cv_means = None

        if do_tune:
            grid = config.PARAM_GRIDS.get(model_name, None)
            if grid:
                try:
                    gs = GridSearchCV(pipe, grid, scoring=config.SCORING, cv=3, n_jobs=-1, refit=True, error_score=np.nan)
                    gs.fit(X_tr, y_tr)
                    if hasattr(gs, "best_estimator_") and gs.best_estimator_ is not None:
                        pipe = gs.best_estimator_
                except Exception:
                    pass

        pipe.fit(X_tr, y_tr)
        y_proba = _get_proba(pipe, X_val); y_pred = (y_proba>=0.5).astype(int)

        opt_thr = best_threshold(y_val, y_proba, mode=thr_mode)
        from sklearn.metrics import f1_score
        y_pred_opt = (y_proba>=opt_thr).astype(int)

        metrics = {"model_name": model_name,
                   "roc_auc": float(roc_auc_score(y_val, y_proba)),
                   "accuracy": float(accuracy_score(y_val, y_pred)),
                   "f1": float(f1_score(y_val, y_pred)),
                   "precision": float(precision_score(y_val, y_pred)),
                   "recall": float(recall_score(y_val, y_pred)),
                   "opt_thr": float(opt_thr),
                   "f1_opt": float(f1_score(y_val, y_pred_opt)),
                   "n_samples": int(len(y_val))}

        curves = _save_plots(y_val, y_proba, model_name, tag=artifact_tag or model_name)

        try:
            result = permutation_importance(pipe, X_val, y_val, n_repeats=10, random_state=config.RANDOM_STATE, scoring="roc_auc")
            importances = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)
            perm_path = f"assets/perm_{artifact_tag or model_name}.csv"
            importances.to_csv(perm_path, index=False)
        except Exception:
            perm_path = None

        cand_path = config.TEMP_MODEL_PATH if artifact_tag is None else f"models/candidate_{artifact_tag}.joblib"
        joblib.dump(pipe, cand_path)
        duration = time.perf_counter() - t0
        _log_run(model_name, model_params, metrics, cand_path, duration)

        return {"ok": True, "candidate_path": cand_path, "val_metrics": metrics, "cv_means": cv_means,
                "curves": curves, "params_used": model_params, "perm_csv": perm_path}
    except Exception as e:
        return {"ok": False, "errors": [f"{type(e).__name__}: {e}"]}

def evaluate_model(model_path: str, data_path: str=None, artifact_tag: Optional[str]=None) -> Dict[str, Any]:
    _ensure_dirs()
    if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found: {model_path}")
    pipe = joblib.load(model_path)
    df = load_data(data_path or config.TRAIN_DATA_PATH)
    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    y_proba = _get_proba(pipe, X_te); y_pred = (y_proba>=0.5).astype(int)
    metrics = {"model_name": "loaded_model",
               "roc_auc": float(roc_auc_score(y_te, y_proba)),
               "accuracy": float(accuracy_score(y_te, y_pred)),
               "f1": float(f1_score(y_te, y_pred)),
               "precision": float(precision_score(y_te, y_pred)),
               "recall": float(recall_score(y_te, y_pred)),
               "n_samples": int(len(y_te))}
    curves = _save_plots(y_te, y_proba, "loaded_model", tag=artifact_tag)
    return {"metrics": metrics, "curves": curves}

def promote_model_to_production(src_path: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> str:
    _ensure_dirs()
    src = src_path or config.TEMP_MODEL_PATH
    if not os.path.exists(src): raise FileNotFoundError("No candidate model to promote.")
    shutil.copy(src, config.MODEL_PATH)
    meta = {"model_path": config.MODEL_PATH}
    if metadata: meta.update(metadata)
    Path(config.BEST_META_PATH).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if os.path.exists(src) and src != config.MODEL_PATH:
            os.remove(src)
    except Exception:
        pass
    return f"✅ Promoted to production: {config.MODEL_PATH}"

def has_production() -> bool:
    return os.path.exists(config.MODEL_PATH)

def read_best_meta() -> Dict[str, Any]:
    try:
        return json.loads(Path(config.BEST_META_PATH).read_text(encoding="utf-8"))
    except Exception:
        return {}

def predict_with_production(df_features: pd.DataFrame, threshold: float=0.5) -> pd.DataFrame:
    if not has_production(): raise FileNotFoundError("No production model found. Please promote a model first.")
    pipe = joblib.load(config.MODEL_PATH)
    proba = _get_proba(pipe, df_features)
    pred = (proba>=threshold).astype(int)
    out = df_features.copy(); out["proba_PD"] = proba; out["pred"] = pred
    return out
