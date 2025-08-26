import io, json
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
import model_pipeline as mp

st.set_page_config(page_title="Parkinsons – Pro (v7+)", layout="wide")
st.title("🧪 Parkinsons – ML App (Pro, v7+)")
st.caption("Data & EDA • Single • Multi • Best • Predict • Retrain • Exports")

@st.cache_data
def load_df():
    return mp.load_data(config.TRAIN_DATA_PATH)

def read_csv_flex(file) -> pd.DataFrame:
    for enc in ["utf-8","latin-1","cp1255"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    file.seek(0); return pd.read_csv(file, errors="ignore")

def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    try:
        try:
            import openpyxl  # noqa
            engine = "openpyxl"
        except Exception:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        with pd.ExcelWriter(bio, engine=engine) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        bio.seek(0)
        return bio.read()
    except Exception:
        first_df = next(iter(sheets.values())) if sheets else pd.DataFrame()
        return first_df.to_csv(index=False).encode("utf-8")

def style_best(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return df
    i_best = df[metric].astype(float).idxmax()
    df = df.copy()
    if "model_name" in df.columns:
        df.loc[i_best, "model_name"] = "⭐ " + str(df.loc[i_best, "model_name"])
    return df

df = load_df()
features = config.FEATURES
target = config.TARGET

tab_data, tab_single, tab_multi, tab_best, tab_predict, tab_retrain = st.tabs(
    ["📊 Data & EDA","🎯 Single Model","🏁 Multi Compare","🏆 Best Dashboard","🔮 Predict","🔁 Retrain"]
)

with tab_data:
    st.subheader("Dataset")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(30), use_container_width=True)
    with st.expander("🔎 EDA (expand)", expanded=False):
        left, right = st.columns([1.2,1])
        with left:
            st.write("**Missing values (top 20):**")
            miss_df = (
                df[features + [target]]
                .isna()
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .rename("missing")
                .reset_index()
                .rename(columns={"index":"column"})
            )
            st.dataframe(miss_df, use_container_width=True)
            st.download_button("⬇️ missing.csv", miss_df.to_csv(index=False), "missing.csv", "text/csv")
            st.write("**Descriptive stats:**")
            desc_df = df[features].describe().T
            st.dataframe(desc_df, use_container_width=True)
            st.download_button("⬇️ describe.csv", desc_df.to_csv(), "describe.csv", "text/csv")
        with right:
            st.write("**Class balance:**")
            cls = df[target].value_counts().rename({0:"No-PD", 1:"PD"})
            st.bar_chart(cls)
        st.write("**Correlation heatmap:**")
        corr = df[features + [target]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        if st.button("⬇️ Export EDA to Excel"):
            xls = to_excel_bytes({"missing": miss_df, "describe": desc_df, "corr": corr.reset_index()})
            st.download_button("Download EDA.xlsx", data=xls, file_name="eda_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def edit_params(model_name: str, key_prefix: str=""):
    import config
    params = config.DEFAULT_PARAMS.get(model_name, {}).copy()
    cols = st.columns(3); edited={}; i=0
    for k,v in params.items():
        with cols[i%3]:
            skey = f"{key_prefix}{model_name}_{k}"
            if isinstance(v,bool): edited[k]=st.checkbox(k,value=v,key=skey)
            elif isinstance(v,int): edited[k]=st.number_input(k,value=int(v),step=1,key=skey)
            elif isinstance(v,float): edited[k]=st.number_input(k,value=float(v),key=skey,format="%.6f")
            elif isinstance(v,tuple): edited[k]=st.text_input(k,value=str(v),key=skey)
            else: edited[k]=st.text_input(k,value=str(v),key=skey)
        i+=1
    for k,v in edited.items():
        if isinstance(v,str) and v.startswith("(") and v.endswith(")"):
            try: edited[k]=eval(v)
            except Exception: pass
    return edited

with tab_single:
    st.subheader("Train & Evaluate a single model (on current EDA dataset)")
    chosen = st.selectbox("Choose model", config.MODEL_LIST, index=config.MODEL_LIST.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in config.MODEL_LIST else 0)
    colA, colB, colC, colD = st.columns(4)
    with colA: do_cv = st.checkbox("Cross-Validation", True)
    with colB: do_tune = st.checkbox("GridSearch", True)
    with colC: use_groups = st.checkbox("Prevent leakage (Group by 'name')", True)
    with colD: use_smote = st.checkbox("SMOTE", False)
    calibrate = st.checkbox("Calibrate (isotonic)", False)
    thr_mode = st.selectbox("Threshold strategy", ["youden","f1"], index=0)
    params = edit_params(chosen, "single_")
    if st.button("🚀 Train model", key="single_train"):
        res = mp.train_model(config.TRAIN_DATA_PATH, chosen, params, do_cv=do_cv, do_tune=do_tune,
                             artifact_tag=f"single_{chosen}", use_groups=use_groups, use_smote=use_smote,
                             calibrate=calibrate, thr_mode=thr_mode)
        if not res.get("ok"):
            st.error("\\n".join(res.get("errors", [])))
        else:
            st.success(f"Candidate saved: {res['candidate_path']}")
            mets = res["val_metrics"]; df_m = pd.DataFrame([mets])
            st.dataframe(df_m, use_container_width=True)
            cv_means = res.get("cv_means")
            if cv_means:
                st.markdown("**Cross-Validation (mean over folds):**")
                st.dataframe(pd.DataFrame([cv_means]), use_container_width=True)
            col1,col2,col3 = st.columns(3)
            p = Path(res["curves"]["roc_path"])
            if p.exists(): col1.image(str(p), caption="ROC")
            p = Path(res["curves"]["pr_path"])
            if p.exists(): col2.image(str(p), caption="PR")
            p = Path(res["curves"]["cm_path"])
            if p.exists(): col3.image(str(p), caption="Confusion Matrix")
            perm_csv = res.get("perm_csv")
            if perm_csv and Path(perm_csv).exists():
                imp_df = pd.read_csv(perm_csv).head(20)
                st.markdown("**Top permutation importances:**")
                st.dataframe(imp_df, use_container_width=True)

with tab_multi:
    st.subheader("Train & Compare multiple models (on current EDA dataset)")
    pick = st.multiselect("Select models", options=config.MODEL_LIST, default=["XGBoost","RandomForest","LogisticRegression"])
    do_cv2 = st.checkbox("Cross-Validation", True, key="multi_cv")
    do_tune2 = st.checkbox("GridSearch", True, key="multi_tune")
    use_groups2 = st.checkbox("Prevent leakage (Group by 'name')", True, key="multi_groups")
    use_smote2 = st.checkbox("SMOTE", False, key="multi_smote")
    calibrate2 = st.checkbox("Calibrate", False, key="multi_calib")
    thr_mode2 = st.selectbox("Threshold", ["youden","f1"], index=0, key="multi_thr")
    param_map={}
    for m in pick:
        with st.expander(f"⚙️ Parameters — {m}", expanded=False):
            param_map[m] = edit_params(m, f"multi_{m}_")
    if st.button("🏁 Train & Compare", key="multi_train"):
        leaderboard=[]; roc_curves={}; pr_curves={}
        for m in pick:
            res = mp.train_model(config.TRAIN_DATA_PATH, m, param_map.get(m, {}), do_cv=do_cv2, do_tune=do_tune2,
                                 artifact_tag=f"multi_{m}", use_groups=use_groups2, use_smote=use_smote2,
                                 calibrate=calibrate2, thr_mode=thr_mode2)
            if res.get("ok"):
                row = res["val_metrics"].copy(); row["model_name"]=m; row["candidate_path"]=res["candidate_path"]; row["params"]=json.dumps(res.get("params_used", param_map.get(m, {})))
                leaderboard.append(row)
                roc_curves[m] = {"fpr": res["curves"]["fpr"], "tpr": res["curves"]["tpr"]}
                pr_curves[m] = {"prec": res["curves"]["prec"], "rec": res["curves"]["rec"]}
        if leaderboard:
            df_lb = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
            i_best = df_lb["roc_auc"].astype(float).idxmax()
            df_lb.loc[i_best, "model_name"] = "⭐ " + str(df_lb.loc[i_best, "model_name"])
            st.dataframe(df_lb, use_container_width=True)
            metric_choice = st.selectbox("Metric for bar chart", ["roc_auc","accuracy","f1","precision","recall"], index=0)
            st.bar_chart(pd.DataFrame(df_lb.set_index("model_name")[metric_choice]))
            figR, axR = plt.subplots(figsize=(5.5,4))
            for name, c in roc_curves.items(): axR.plot(c["fpr"], c["tpr"], label=name)
            axR.plot([0,1],[0,1],"--", lw=0.7); axR.set_xlabel("FPR"); axR.set_ylabel("TPR"); axR.set_title("ROC Curves"); axR.legend()
            st.pyplot(figR)
            figP, axP = plt.subplots(figsize=(5.5,4))
            for name, c in pr_curves.items(): axP.plot(c["rec"], c["prec"], label=name)
            axP.set_xlabel("Recall"); axP.set_ylabel("Precision"); axP.set_title("PR Curves"); axP.legend()
            st.pyplot(figP)
            st.download_button("⬇️ leaderboard.csv", df_lb.to_csv(index=False), "leaderboard.csv", "text/csv")
        else:
            st.warning("No models trained.")

with tab_best:
    st.subheader("Best Model Dashboard (pre-trained baseline + history)")
    if mp.has_production():
        meta = mp.read_best_meta()
        st.write("**Best model metadata:**"); st.json(meta)
        try:
            ev = mp.evaluate_model(config.MODEL_PATH, artifact_tag="best_eval")
            mets = ev["metrics"]
            st.dataframe(pd.DataFrame([mets]))
            st.download_button("⬇️ best_eval_metrics.csv", pd.DataFrame([mets]).to_csv(index=False), "best_eval_metrics.csv", "text/csv")
            for p,cap in [("assets/roc_best_eval.png","ROC"),("assets/pr_best_eval.png","PR"),("assets/cm_best_eval.png","Confusion Matrix")]:
                if Path(p).exists(): st.image(p, caption=cap)
        except Exception as e:
            st.error(str(e))
        if Path(config.RUNS_CSV).exists():
            runs = pd.read_csv(config.RUNS_CSV)
            st.markdown("**Experiment history (runs.csv):**")
            st.dataframe(runs.tail(50), use_container_width=True)
            st.download_button("⬇️ runs.csv", runs.to_csv(index=False), "runs.csv", "text/csv")
    else:
        st.warning("No production model yet.")

with tab_predict:
    st.subheader("Predict with the current best model")
    if not mp.has_production():
        st.warning("No production model found.")
    else:
        default_thr = 0.5
        meta = mp.read_best_meta()
        if "opt_thr" in meta: default_thr = float(meta["opt_thr"])
        thr = st.slider("Decision threshold", 0.0, 1.0, value=float(default_thr), step=0.01)
        feats = df[features]
        preds = mp.predict_with_production(feats, threshold=thr)
        st.write("Preview predictions (first 20):"); st.dataframe(preds.head(20), use_container_width=True)
        st.download_button("⬇️ predictions.csv", preds.to_csv(index=False), "predictions.csv", "text/csv")
        st.markdown("### Single patient input")
        cols = st.columns(3)
        single_vals = {}
        for i, f in enumerate(features):
            with cols[i % 3]:
                default = float(df[f].median()) if f in df.columns else 0.0
                single_vals[f] = st.number_input(f, value=default, format="%.6f", key=f"single_{f}")
        if st.button("Predict single"):
            row = pd.DataFrame([single_vals])
            out = mp.predict_with_production(row, threshold=thr)
            label = "PD" if int(out.loc[0, "pred"]) == 1 else "No-PD"
            prob = float(out.loc[0, "proba_PD"])
            st.success(f"Prediction: **{label}** (p={prob:.3f})")
            st.dataframe(out, use_container_width=True)
        st.markdown("### Upload CSV for batch prediction")
        up = st.file_uploader("CSV with *feature* columns only (no 'name'/'status')", type=["csv"], key="pred_batch")
        if st.button("Run batch predictions"):
            if up is None:
                st.error("Please upload a CSV.")
            else:
                try:
                    df_in = read_csv_flex(up)
                    out = mp.predict_with_production(df_in[features], threshold=thr)
                    st.dataframe(out.head(30), use_container_width=True)
                    st.download_button("⬇️ predictions_batch.csv", out.to_csv(index=False), "predictions_batch.csv", "text/csv")
                except Exception as e:
                    st.error(str(e))

with tab_retrain:
    st.subheader("Retrain with additional data (optional) — compare vs current best")
    st.caption("Upload CSV with the same schema (include all feature columns + target 'status').")
    up_new = st.file_uploader("Upload training CSV", type=["csv"], key="train_new")
    model_r = st.selectbox("Choose model", config.MODEL_LIST, index=config.MODEL_LIST.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in config.MODEL_LIST else 0, key="new_model_sel")
    def edit_preset(model_name: str):
        import config
        params = config.DEFAULT_PARAMS.get(model_name, {}).copy()
        cols = st.columns(3); edited={}; i=0
        for k,v in params.items():
            with cols[i%3]:
                skey = f"re_{model_name}_{k}"
                if isinstance(v,bool): edited[k]=st.checkbox(k,value=v,key=skey)
                elif isinstance(v,int): edited[k]=st.number_input(k,value=int(v),step=1,key=skey)
                elif isinstance(v,float): edited[k]=st.number_input(k,value=float(v),key=skey,format="%.6f")
                elif isinstance(v,tuple): edited[k]=st.text_input(k,value=str(v),key=skey)
                else: edited[k]=st.text_input(k,value=str(v),key=skey)
            i+=1
        for k,v in edited.items():
            if isinstance(v,str) and v.startswith("(") and v.endswith(")"):
                try: edited[k]=eval(v)
                except Exception: pass
        return edited
    params_r = edit_preset(model_r)
    use_groups_r = st.checkbox("Prevent leakage (Group by 'name')", True, key="re_groups")
    use_smote_r = st.checkbox("SMOTE", False, key="re_smote")
    calibrate_r = st.checkbox("Calibrate", False, key="re_calib")
    thr_mode_r = st.selectbox("Threshold", ["youden","f1"], index=0, key="re_thr")
    if st.button("🛠️ Train on uploaded data"):
        if up_new is None:
            st.error("Please upload a CSV.")
        else:
            try:
                df_new = read_csv_flex(up_new)
                tmp_path = "data/_uploaded_train.csv"
                df_new.to_csv(tmp_path, index=False)
                res_new = mp.train_model(tmp_path, model_name=model_r, model_params=params_r, do_cv=True, do_tune=True,
                                         artifact_tag=f"upload_{model_r}", use_groups=use_groups_r, use_smote=use_smote_r,
                                         calibrate=calibrate_r, thr_mode=thr_mode_r)
                if not res_new.get("ok"):
                    st.error("\\n".join(res_new.get("errors", [])))
                else:
                    st.success("New candidate trained on uploaded data.")
                    st.json(res_new["val_metrics"])
                    if mp.has_production():
                        mets_best = mp.evaluate_model(config.MODEL_PATH, data_path=tmp_path, artifact_tag="prod_eval")["metrics"]
                        st.write("Production metrics on the same uploaded data:"); st.json(mets_best)
                    if st.button("⭐ Promote uploaded-data model as new best"):
                        meta = {"source": "retrain_upload", "model_name": model_r, "metrics": res_new["val_metrics"], "params": res_new.get("params_used", params_r)}
                        if "opt_thr" in res_new["val_metrics"]:
                            meta["opt_thr"] = float(res_new["val_metrics"]["opt_thr"])
                        msg = mp.promote_model_to_production(f"models/candidate_upload_{model_r}.joblib", metadata=meta)
                        st.success(msg)
            except Exception as e:
                st.error(str(e))

st.markdown("---")
st.caption("v7+: EDA exports • Single & Multi tables/graphs • Best dashboard history • Single/batch prediction • Retrain with promotion")
