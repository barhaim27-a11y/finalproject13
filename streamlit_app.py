
import io, json
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
import model_pipeline as mp

st.set_page_config(page_title="Parkinsons - Pro (v7+ full)", layout="wide")
st.title("Parkinsons - ML App (Pro, v7+ full)")
st.caption("Data & EDA | Single | Multi | Best | Predict | Retrain | Exports")

# Helpers
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

def to_excel_bytes(sheets: dict) -> bytes:
    # Return XLSX bytes with multiple sheets; fallback to CSV bytes of first sheet.
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
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df.to_excel(writer, sheet_name=(name or "Sheet")[:31], index=False)
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
        df.loc[i_best, "model_name"] = "STAR " + str(df.loc[i_best, "model_name"])
    return df

# Data
df = load_df()
features = config.FEATURES
target = config.TARGET

tab_data, tab_single, tab_multi, tab_best, tab_predict, tab_retrain = st.tabs(
    ["DATA/EDA","Single Model","Multi Compare","Best Dashboard","Predict","Retrain"]
)

# EDA
with tab_data:
    st.subheader("Dataset")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(30), use_container_width=True)

    with st.expander("EDA (expand)", expanded=False):
        left, right = st.columns([1.2,1])
        with left:
            st.write("Missing values (top 20):")
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
            st.download_button("missing.csv", miss_df.to_csv(index=False), "missing.csv", "text/csv")

            st.write("Descriptive stats:")
            desc_df = df[features].describe().T
            st.dataframe(desc_df, use_container_width=True)
            st.download_button("describe.csv", desc_df.to_csv(), "describe.csv", "text/csv")

        with right:
            st.write("Class balance:")
            cls = df[target].value_counts().rename({0:"No-PD", 1:"PD"})
            st.bar_chart(cls)

        st.write("Correlation heatmap:")
        corr = df[features + [target]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

        xls = to_excel_bytes({"missing": miss_df, "describe": desc_df, "corr": corr.reset_index()})
        st.download_button("Export EDA.xlsx", data=xls, file_name="eda_export.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="eda_xlsx")

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

# Single
with tab_single:
    st.subheader("Train & Evaluate a single model (on current EDA dataset)")
    chosen = st.selectbox("Choose model", config.MODEL_LIST, index=config.MODEL_LIST.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in config.MODEL_LIST else 0)
    colA, colB, colC, colD = st.columns(4)
    with colA: do_cv = st.checkbox("Cross-Validation", True)
    with colB: do_tune = st.checkbox("GridSearch", True)
    with colC: use_groups = st.checkbox("Group by 'name'", True)
    with colD: use_smote = st.checkbox("SMOTE", False)
    calibrate = st.checkbox("Calibrate (isotonic)", False)
    thr_mode = st.selectbox("Threshold strategy", ["youden","f1"], index=0)
    params = edit_params(chosen, "single_")
    if st.button("Train model", key="single_train"):
        res = mp.train_model(config.TRAIN_DATA_PATH, chosen, params, do_cv=do_cv, do_tune=do_tune,
                             artifact_tag=f"single_{chosen}", use_groups=use_groups, use_smote=use_smote,
                             calibrate=calibrate, thr_mode=thr_mode)
        if not res.get("ok"):
            st.error("\n".join(res.get("errors", [])))
        else:
            st.success(f"Candidate saved: {res['candidate_path']}")
            mets = res["val_metrics"]; df_m = pd.DataFrame([mets])
            st.markdown("Metrics:"); st.dataframe(df_m, use_container_width=True)
            cv_means = res.get("cv_means")
            if cv_means:
                st.markdown("Cross-Validation (mean over folds):")
                st.dataframe(pd.DataFrame([cv_means]), use_container_width=True)
            col1,col2,col3 = st.columns(3)
            p = Path(res["curves"]["roc_path"])
            if p.exists(): col1.image(str(p), caption="ROC")
            p = Path(res["curves"]["pr_path"])
            if p.exists(): col2.image(str(p), caption="PR")
            p = Path(res["curves"]["cm_path"])
            if p.exists(): col3.image(str(p), caption="Confusion Matrix")
            perm_csv = res.get("perm_csv")
            imp_df = None
            if perm_csv and Path(perm_csv).exists():
                imp_df = pd.read_csv(perm_csv).head(20)
                st.markdown("Top permutation importances:")
                st.dataframe(imp_df, use_container_width=True)
            xls = to_excel_bytes({"metrics": df_m, "cv_means": pd.DataFrame([cv_means]) if cv_means else pd.DataFrame(), "perm_importance": imp_df if isinstance(imp_df, pd.DataFrame) else pd.DataFrame()})
            st.download_button("Export Single Results (XLSX)", xls, file_name=f"single_{chosen}_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="single_xlsx")

# Multi
with tab_multi:
    st.subheader("Train & Compare multiple models (on current EDA dataset)")
    pick = st.multiselect("Select models", options=config.MODEL_LIST, default=["XGBoost","RandomForest","LogisticRegression"])
    do_cv2 = st.checkbox("Cross-Validation", True, key="multi_cv")
    do_tune2 = st.checkbox("GridSearch", True, key="multi_tune")
    use_groups2 = st.checkbox("Group by 'name'", True, key="multi_groups")
    use_smote2 = st.checkbox("SMOTE", False, key="multi_smote")
    calibrate2 = st.checkbox("Calibrate", False, key="multi_calib")
    thr_mode2 = st.selectbox("Threshold", ["youden","f1"], index=0, key="multi_thr")
    param_map={}
    for m in pick:
        with st.expander(f"Parameters - {m}", expanded=False):
            param_map[m] = edit_params(m, f"multi_{m}_")
    if st.button("Train & Compare", key="multi_train"):
        leaderboard=[]; roc_curves={}; pr_curves={}
        for m in pick:
            res = mp.train_model(config.TRAIN_DATA_PATH, m, param_map.get(m, {}), do_cv=do_cv2, do_tune=do_tune2,
                                 artifact_tag=f"multi_{m}", use_groups=use_groups2, use_smote=use_smote2,
                                 calibrate=calibrate2, thr_mode=thr_mode2)
            if res.get("ok"):
                row = res["val_metrics"].copy()
                row["model_name"]=m
                row["candidate_path"]=res["candidate_path"]
                row["params"]=json.dumps(res.get("params_used", param_map.get(m, {})))
                leaderboard.append(row)
                roc_curves[m] = {"fpr": res["curves"]["fpr"], "tpr": res["curves"]["tpr"]}
                pr_curves[m] = {"prec": res["curves"]["prec"], "rec": res["curves"]["rec"]}
        if leaderboard:
            df_lb = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
            i_best = df_lb["roc_auc"].astype(float).idxmax()
            df_lb.loc[i_best, "model_name"] = "STAR " + str(df_lb.loc[i_best, "model_name"])
            st.dataframe(df_lb, use_container_width=True)
            xls = to_excel_bytes({"leaderboard": df_lb})
            st.download_button("Export Leaderboard (XLSX)", xls, file_name="leaderboard.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="multi_xlsx")
            st.download_button("leaderboard.csv", df_lb.to_csv(index=False), "leaderboard.csv", "text/csv", key="multi_csv")
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
        else:
            st.warning("No models trained.")

# Best
with tab_best:
    st.subheader("Best Model Dashboard (pre-trained baseline + history)")
    if mp.has_production():
        meta = mp.read_best_meta()
        st.write("Best model metadata:"); st.json(meta)
        try:
            ev = mp.evaluate_model(config.MODEL_PATH, artifact_tag="best_eval")
            mets = ev["metrics"]
            st.dataframe(pd.DataFrame([mets]))
            st.download_button("best_eval_metrics.csv", pd.DataFrame([mets]).to_csv(index=False), "best_eval_metrics.csv", "text/csv", key="best_csv")
            for p,cap in [("assets/roc_best_eval.png","ROC"),("assets/pr_best_eval.png","PR"),("assets/cm_best_eval.png","Confusion Matrix")]:
                if Path(p).exists(): st.image(p, caption=cap)
        except Exception as e:
            st.error(str(e))
        if Path(config.RUNS_CSV).exists():
            runs = pd.read_csv(config.RUNS_CSV)
            st.markdown("Experiment history (runs.csv):")
            st.dataframe(runs.tail(200), use_container_width=True)
            st.download_button("runs.csv", runs.to_csv(index=False), "runs.csv", "text/csv", key="runs_csv")
            if "model" in runs.columns:
                lb_cols = ["model"] + [c for c in runs.columns if c.startswith("metric_")]
                last_by_model = runs.groupby("model").tail(1)[lb_cols]
                rename_map = {c: c.replace("metric_", "") for c in lb_cols if c.startswith("metric_")}
                last_by_model = last_by_model.rename(columns=rename_map).rename(columns={"model":"model_name"})
                if "roc_auc" in last_by_model.columns:
                    last_by_model = last_by_model.sort_values("roc_auc", ascending=False).reset_index(drop=True)
                    last_by_model = style_best(last_by_model, "roc_auc")
                st.markdown("Last results per model:")
                st.dataframe(last_by_model, use_container_width=True)
                xls = to_excel_bytes({"best_dashboard_leaderboard": last_by_model})
                st.download_button("Export Best Dashboard (XLSX)", xls, file_name="best_dashboard.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="best_xlsx")
    else:
        st.warning("No production model yet.")

# Predict
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
        st.download_button("predictions.csv", preds.to_csv(index=False), "predictions.csv", "text/csv", key="pred_csv")
        st.markdown("Single patient input")
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
            st.success(f"Prediction: {label} (p={prob:.3f})")
            st.dataframe(out, use_container_width=True)
            xls = to_excel_bytes({"single_prediction": out})
            st.download_button("Export single prediction (XLSX)", xls, "single_prediction.xlsx", key="single_pred_xlsx")
        st.markdown("Upload CSV for batch prediction")
        up = st.file_uploader("CSV with feature columns only (no 'name'/'status')", type=["csv"], key="pred_batch")
        if st.button("Run batch predictions"):
            if up is None:
                st.error("Please upload a CSV.")
            else:
                try:
                    df_in = read_csv_flex(up)
                    out = mp.predict_with_production(df_in[features], threshold=thr)
                    st.dataframe(out.head(30), use_container_width=True)
                    xls = to_excel_bytes({"predictions_batch": out})
                    st.download_button("Export predictions (XLSX)", xls, "predictions_batch.xlsx", key="pred_batch_xlsx")
                    st.download_button("predictions_batch.csv", out.to_csv(index=False), "predictions_batch.csv", "text/csv", key="pred_batch_csv")
                except Exception as e:
                    st.error(str(e))

# Retrain
with tab_retrain:
    st.subheader("Retrain with additional data - compare vs current best")
    st.caption("Upload CSV with the same schema (include all feature columns + target 'status').")
    up_new = st.file_uploader("Upload training CSV", type=["csv"], key="train_new")
    st.markdown("Retrain a single model")
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
    use_groups_r = st.checkbox("Group by 'name'", True, key="re_groups")
    use_smote_r = st.checkbox("SMOTE", False, key="re_smote")
    calibrate_r = st.checkbox("Calibrate", False, key="re_calib")
    thr_mode_r = st.selectbox("Threshold", ["youden","f1"], index=0, key="re_thr")
    if st.button("Train single on uploaded data"):
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
                    st.error("\n".join(res_new.get("errors", [])))
                else:
                    st.success("New candidate trained on uploaded data.")
                    st.json(res_new["val_metrics"])
                    if mp.has_production():
                        mets_best = mp.evaluate_model(config.MODEL_PATH, data_path=tmp_path, artifact_tag="prod_eval")["metrics"]
                        st.write("Production metrics on the same uploaded data:"); st.json(mets_best)
                        comp = pd.DataFrame([res_new["val_metrics"], mets_best], index=["new_candidate", "production"])
                        xls = to_excel_bytes({"compare_new_vs_prod": comp})
                        st.download_button("Export compare (XLSX)", xls, "compare_new_vs_prod.xlsx", key="re_single_xlsx")
                    if st.button("Promote this model as new best"):
                        meta = {"source": "retrain_upload_single", "model_name": model_r, "metrics": res_new["val_metrics"], "params": res_new.get("params_used", params_r)}
                        if "opt_thr" in res_new["val_metrics"]:
                            meta["opt_thr"] = float(res_new["val_metrics"]["opt_thr"])
                        msg = mp.promote_model_to_production(f"models/candidate_upload_{model_r}.joblib", metadata=meta)
                        st.success(msg)
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("Train & Compare multiple models on uploaded data")
    pick_r = st.multiselect("Select models", options=config.MODEL_LIST, default=["XGBoost","RandomForest","LogisticRegression"], key="re_multi_pick")
    param_map_r = {}
    for m in pick_r:
        with st.expander(f"Parameters - {m} (retrain)", expanded=False):
            param_map_r[m] = edit_params(m, f"re_multi_{m}_")
    if st.button("Train & Compare (uploaded data)"):
        if up_new is None:
            st.error("Please upload a CSV.")
        else:
            try:
                df_new = read_csv_flex(up_new)
                tmp_path = "data/_uploaded_train.csv"
                df_new.to_csv(tmp_path, index=False)
                leaderboard=[]; candidate_paths={}
                for m in pick_r:
                    res = mp.train_model(tmp_path, model_name=m, model_params=param_map_r.get(m, {}), do_cv=True, do_tune=True,
                                         artifact_tag=f"upload_{m}", use_groups=use_groups_r, use_smote=use_smote_r,
                                         calibrate=calibrate_r, thr_mode=thr_mode_r)
                    if res.get("ok"):
                        row = res["val_metrics"].copy(); row["model_name"]=m; row["candidate_path"]=res["candidate_path"]; row["params"]=json.dumps(res.get("params_used", param_map_r.get(m, {})))
                        leaderboard.append(row)
                        candidate_paths[m] = res["candidate_path"]
                if leaderboard:
                    df_lb = pd.DataFrame(leaderboard).sort_values("roc_auc", ascending=False).reset_index(drop=True)
                    df_lb = style_best(df_lb, "roc_auc")
                    st.dataframe(df_lb, use_container_width=True)
                    if mp.has_production():
                        mets_best = mp.evaluate_model(config.MODEL_PATH, data_path=tmp_path, artifact_tag="prod_eval_multi")["metrics"]
                        st.write("Production metrics on uploaded data:"); st.json(mets_best)
                    xls = to_excel_bytes({"retrain_leaderboard": df_lb})
                    st.download_button("Export retrain leaderboard (XLSX)", xls, "retrain_leaderboard.xlsx", key="re_multi_xlsx")
                    top_name = df_lb.iloc[0]["model_name"].replace("STAR ","")
                    if st.button(f"Promote top candidate ({top_name}) as new best"):
                        metrics_cols = [c for c in ["roc_auc","accuracy","f1","precision","recall","opt_thr","f1_opt","n_samples"] if c in df_lb.columns]
                        top_metrics = json.loads(df_lb.iloc[0][metrics_cols].to_json())
                        msg = mp.promote_model_to_production(candidate_paths[top_name], metadata={"source": "retrain_upload_multi", "model_name": top_name, "metrics": top_metrics, "params": json.loads(df_lb.iloc[0]["params"])})
                        st.success(msg)
                else:
                    st.warning("No models trained.")
            except Exception as e:
                st.error(str(e))

st.markdown("---")
st.caption("v7+ full: EDA exports | Single & Multi results tables/graphs | Best dashboard history | Single/batch prediction | Retrain single/multi with export & promotion")
