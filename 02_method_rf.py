"""
02_method_rf.py  —  MÉTODO 2: RANDOM FOREST SOBRE TENSOR APLANADO

Lee: data/processed/tensor_thec_d.pkl  (generado por 00_build_tensor_pkl.py)

Random Forest sobre el tensor THEC-D aplanado a vector de 256 features.
El clasificador aprende límites de decisión no lineales sobre los features.

Resultados guardados en: results/02_rf_results.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

PKL_PATH    = Path("data/processed/tensor_thec_d.pkl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
N_ESTIMATORS = 300


def load_data():
    print(f"📥 Cargando: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        d = pickle.load(f)
    X  = d["X"].reshape(len(d["X"]), -1)   # aplanar (N, 256)
    y  = d["y"]
    class_names  = d["class_names"]
    feature_names = []
    for seq in d["seq_names"]:
        for feat in d["feature_names"]:
            for w in range(d["num_windows"]):
                feature_names.append(f"{seq}_{feat}_w{w}")
    print(f"   X aplanado: {X.shape}  |  Features: {X.shape[1]}")
    return X, y, class_names, d["random_state"], d["test_size"], feature_names


def main():
    print("=" * 65)
    print("MÉTODO 2 — RANDOM FOREST SOBRE TENSOR APLANADO")
    print("=" * 65 + "\n")

    X, y, class_names, random_state, test_size, feature_names = load_data()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"   Train: {len(X_tr):,}  |  Test: {len(X_te):,}\n")

    print(f"🔄 Entrenando Random Forest ({N_ESTIMATORS} árboles)...")
    t0  = datetime.now()
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    t_train = (datetime.now() - t0).total_seconds()
    print(f"   ✅ Listo en {t_train:.2f}s")

    t0     = datetime.now()
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    t_inf  = (datetime.now() - t0).total_seconds() * 1000 / len(X_te)

    acc    = float(accuracy_score(y_te, y_pred))
    f1_w   = float(f1_score(y_te, y_pred, average="weighted"))
    f1_mac = float(f1_score(y_te, y_pred, average="macro"))
    f1_mic = float(f1_score(y_te, y_pred, average="micro"))
    y_bin  = label_binarize(y_te, classes=list(range(len(class_names))))
    try:
        auc_mac = float(roc_auc_score(y_bin, y_proba, average="macro",    multi_class="ovr"))
        auc_w   = float(roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovr"))
    except Exception:
        auc_mac = auc_w = 0.0

    print(f"\n{'='*65}")
    print("RESULTADOS GLOBALES — RANDOM FOREST")
    print(f"{'='*65}")
    print(f"   Accuracy:              {acc*100:.4f}%")
    print(f"   F1 weighted:           {f1_w*100:.4f}%")
    print(f"   F1 macro:              {f1_mac*100:.4f}%")
    print(f"   F1 micro:              {f1_mic*100:.4f}%")
    print(f"   AUC-ROC macro:         {auc_mac:.4f}")
    print(f"   AUC-ROC weighted:      {auc_w:.4f}")
    print(f"   Latencia/muestra:      {t_inf:.4f} ms")
    print(f"   Tiempo entrenamiento:  {t_train:.4f} s")
    print(f"   Árboles:               {N_ESTIMATORS}")

    cr = classification_report(y_te, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{'='*65}")
    print("CLASSIFICATION REPORT POR CLASE")
    print(f"{'='*65}")
    print(classification_report(y_te, y_pred, target_names=class_names))

    cm = confusion_matrix(y_te, y_pred)
    print(f"{'='*65}")
    print("CONFUSION MATRIX")
    print(f"{'='*65}")
    header = "".join([f"{n[:6]:>8}" for n in [c[:6] for c in class_names]])
    print(f"{'':>22}{header}")
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:20]:<20}" + "".join([f"{v:>8}" for v in row]))

    print(f"\n{'='*65}")
    print("AUC-ROC POR CLASE")
    print(f"{'='*65}")
    auc_per_class = {}
    for i, name in enumerate(class_names):
        try:
            auc = float(roc_auc_score(y_bin[:,i], y_proba[:,i]))
        except Exception:
            auc = 0.0
        auc_per_class[name] = auc
        flag = " ⚠️" if auc < 0.85 else ""
        print(f"   {name:<25} {auc:.4f}{flag}")
    print(f"   {'MACRO AVERAGE':<25} {auc_mac:.4f}")

    # Importancia de features — agrupada por componente y tipo
    print(f"\n{'='*65}")
    print("IMPORTANCIA DE FEATURES (agrupada por dimensión del tensor)")
    print(f"{'='*65}")
    imp = clf.feature_importances_.reshape(4, 8, 8)

    seq_names  = ["I0", "I1", "I2", "I2/I0"]
    feat_names = ["RMS","Entropy","E_2harm","Skewness",
                  "Slope","Accel","Relative","Curvature"]

    print(f"\n  Por componente de secuencia:")
    imp_by_seq = {}
    for s, name in enumerate(seq_names):
        v = float(np.sum(imp[s]))
        imp_by_seq[name] = v
        print(f"    {name:<10}: {v:.4f}  ({v*100:.1f}%)")

    print(f"\n  Por tipo de feature:")
    imp_by_feat = {}
    for f_idx, name in enumerate(feat_names):
        v   = float(np.sum(imp[:, f_idx, :]))
        tag = " ← inter-ventana" if f_idx >= 4 else ""
        imp_by_feat[name] = v
        print(f"    {name:<14}: {v:.4f}  ({v*100:.1f}%){tag}")

    intra_total = sum(imp_by_feat[feat_names[i]] for i in range(4))
    inter_total = sum(imp_by_feat[feat_names[i]] for i in range(4, 8))
    print(f"\n  Intra-ventana total: {intra_total:.4f} ({intra_total*100:.1f}%)")
    print(f"  Inter-ventana total: {inter_total:.4f} ({inter_total*100:.1f}%)")
    print(f"  → Si inter > 25%: la dinámica temporal justifica el tensor")

    # Top 10 features individuales
    print(f"\n  Top 10 features individuales más importantes:")
    flat_imp = clf.feature_importances_
    top10    = np.argsort(flat_imp)[::-1][:10]
    for rank, idx in enumerate(top10):
        s_idx  = idx // (8*8)
        f_idx  = (idx // 8) % 8
        w_idx  = idx % 8
        print(f"    {rank+1:2d}. {seq_names[s_idx]}/{feat_names[f_idx]}/ventana{w_idx}: "
              f"{flat_imp[idx]:.4f}")

    # Espira vs Monofásica-A
    print(f"\n{'='*65}")
    print("ANÁLISIS ESPECÍFICO — ESPIRA-ESPIRA vs MONOFÁSICA-A")
    print(f"{'='*65}")
    mask_bin = (y_te == 0) | (y_te == 11)
    acc_bin  = float(accuracy_score(y_te[mask_bin], y_pred[mask_bin]))
    print(f"   Accuracy binario (solo estas 2 clases): {acc_bin*100:.2f}%")
    print(f"   Support Monofásica-A: {(y_te == 0).sum()}")
    print(f"   Support Espira-Espira: {(y_te == 11).sum()}")

    # Cross-validation
    print(f"\n{'='*65}")
    print("CROSS-VALIDATION 5-FOLD")
    print(f"{'='*65}")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_acc, cv_f1 = [], []
    for fold, (tr_i, te_i) in enumerate(skf.split(X, y)):
        clf_cv = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                        random_state=random_state, n_jobs=-1,
                                        class_weight="balanced")
        clf_cv.fit(X[tr_i], y[tr_i])
        yp = clf_cv.predict(X[te_i])
        cv_acc.append(accuracy_score(y[te_i], yp))
        cv_f1.append(f1_score(y[te_i], yp, average="weighted"))
        print(f"   Fold {fold+1}: Acc={cv_acc[-1]*100:.2f}%  F1={cv_f1[-1]*100:.2f}%")
    print(f"   MEDIA: Acc={np.mean(cv_acc)*100:.2f}% ± {np.std(cv_acc)*100:.2f}%  "
          f"F1={np.mean(cv_f1)*100:.2f}% ± {np.std(cv_f1)*100:.2f}%")

    output = {
        "method": "Random Forest",
        "timestamp": datetime.now().isoformat(),
        "dataset": {"total": int(len(X)), "train": int(len(X_tr)), "test": int(len(X_te))},
        "model": {"n_estimators": N_ESTIMATORS, "train_time_s": t_train,
                  "inference_ms_per_sample": t_inf},
        "global_metrics": {"accuracy": acc, "f1_weighted": f1_w,
                           "f1_macro": f1_mac, "f1_micro": f1_mic,
                           "auc_roc_macro": auc_mac, "auc_roc_weighted": auc_w},
        "per_class_metrics": cr,
        "auc_roc_per_class": auc_per_class,
        "confusion_matrix": cm.tolist(),
        "feature_importance": {
            "by_sequence": imp_by_seq,
            "by_feature_type": imp_by_feat,
            "intra_window_total": intra_total,
            "inter_window_total": inter_total,
        },
        "binary_espira_monofasica": {"accuracy": acc_bin},
        "cross_validation": {
            "acc_mean": float(np.mean(cv_acc)), "acc_std": float(np.std(cv_acc)),
            "f1_mean":  float(np.mean(cv_f1)),  "f1_std":  float(np.std(cv_f1)),
            "folds_acc": [float(v) for v in cv_acc],
            "folds_f1":  [float(v) for v in cv_f1],
        },
    }
    out_path = RESULTS_DIR / "02_rf_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 {out_path}")


if __name__ == "__main__":
    main()