"""
01_method_frobenius.py  —  MÉTODO 1: DISTANCIA DE FROBENIUS TENSORIAL

Lee: data/processed/tensor_thec_d.pkl  (generado por 00_build_tensor_pkl.py)

Clasificador de mínima distancia de Frobenius sobre el tensor THEC-D.
El tensor hace todo el trabajo — 13 parámetros, sin backpropagation.

Distancia de Frobenius: ||T_nueva - T_prototipo_k||_F = sqrt(sum_ijk (t-p)²)
Opera directamente sobre (4,8,8) sin aplanar. Preserva las 3 dimensiones.

Resultados guardados en: results/01_frobenius_results.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")


# RUTAS

PKL_PATH    = Path("data/processed/tensor_thec_d.pkl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# CARGAR DATOS

def load_data():
    print(f"📥 Cargando: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        d = pickle.load(f)
    X            = d["X"]
    y            = d["y"]
    class_names  = d["class_names"]
    random_state = d["random_state"]
    test_size    = d["test_size"]
    tensor_shape = d["tensor_shape"]
    print(f"   X: {X.shape}  |  y: {y.shape}  |  Clases: {len(class_names)}")
    print(f"   Tensor shape: {tensor_shape}  |  Split: {int((1-test_size)*100)}/{int(test_size*100)}")
    return X, y, class_names, random_state, test_size


# CLASIFICADOR DE FROBENIUS

class FrobeniusClassifier:
    """
    Clasificador de mínima distancia de Frobenius tensorial.
    13 parámetros: un tensor prototipo 4×8×8 por clase.
    Sin entrenamiento iterativo — solo promedios de clase.
    """

    def __init__(self):
        self.prototypes_ = {}
        self.classes_    = []

    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y))
        for c in self.classes_:
            self.prototypes_[c] = np.mean(X[y == c], axis=0)
        return self

    def _distances(self, X):
        """Matriz de distancias (N × num_classes)."""
        n    = len(X)
        nc   = len(self.classes_)
        dist = np.zeros((n, nc), dtype=np.float32)
        for j, c in enumerate(self.classes_):
            diff     = X - self.prototypes_[c][np.newaxis]
            dist[:,j] = np.sqrt(np.sum(diff**2, axis=(1,2,3)))
        return dist

    def predict(self, X):
        dist = self._distances(X)
        return np.array([self.classes_[i] for i in np.argmin(dist, axis=1)])

    def predict_proba(self, X):
        """Probabilidad aproximada: softmax inverso de distancias."""
        dist  = self._distances(X)
        inv   = 1.0 / (dist + 1e-10)
        total = inv.sum(axis=1, keepdims=True)
        return inv / total

    def separability_analysis(self, X, y, class_names):
        """
        Análisis de separabilidad por clase:
          intra: distancia promedio de cada muestra a su prototipo correcto
          inter: distancia promedio al prototipo incorrecto más cercano
          ratio: inter/intra (>1 = bien separada, <1 = mezclada)
        """
        dist   = self._distances(X)
        result = {}
        for j, c in enumerate(self.classes_):
            mask  = y == c
            if mask.sum() == 0:
                continue
            intra = float(np.mean(dist[mask, j]))
            others = np.delete(dist[mask], j, axis=1)
            inter  = float(np.mean(np.min(others, axis=1)))
            result[class_names[c]] = {
                "intra_dist": intra,
                "inter_dist": inter,
                "ratio":      inter / (intra + 1e-10),
                "n_samples":  int(mask.sum()),
            }
        return result

    def prototype_distances(self, class_names):
        """Matriz de distancias entre todos los pares de prototipos."""
        nc  = len(self.classes_)
        mat = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                diff     = self.prototypes_[i] - self.prototypes_[j]
                mat[i,j] = float(np.sqrt(np.sum(diff**2)))
        return mat


# MAIN

def main():
    print("=" * 65)
    print("MÉTODO 1 — DISTANCIA DE FROBENIUS TENSORIAL")
    print("=" * 65 + "\n")

    X, y, class_names, random_state, test_size = load_data()
    tensor_shape = X.shape[1:]

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"   Train: {len(X_tr):,}  |  Test: {len(X_te):,}\n")

    # Entrenar
    print("🔄 Entrenando (calculando 13 prototipos)...")
    t0  = datetime.now()
    clf = FrobeniusClassifier()
    clf.fit(X_tr, y_tr)
    t_train = (datetime.now() - t0).total_seconds()
    print(f"   ✅ Listo en {t_train:.2f}s  |  Parámetros: {13 * np.prod(tensor_shape):,}")

    # Predicción
    print("\n🔄 Clasificando test set...")
    t0     = datetime.now()
    y_pred = clf.predict(X_te)
    t_inf  = (datetime.now() - t0).total_seconds() * 1000 / len(X_te)
    y_proba = clf.predict_proba(X_te)

    # Métricas globales
    acc     = float(accuracy_score(y_te, y_pred))
    f1_w    = float(f1_score(y_te, y_pred, average="weighted"))
    f1_mac  = float(f1_score(y_te, y_pred, average="macro"))
    f1_mic  = float(f1_score(y_te, y_pred, average="micro"))

    y_bin   = label_binarize(y_te, classes=list(range(len(class_names))))
    try:
        auc_mac = float(roc_auc_score(y_bin, y_proba, average="macro",    multi_class="ovr"))
        auc_w   = float(roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovr"))
    except Exception:
        auc_mac = auc_w = 0.0

    print(f"\n{'='*65}")
    print(f"RESULTADOS GLOBALES — FROBENIUS TENSORIAL")
    print(f"{'='*65}")
    print(f"   Accuracy:              {acc*100:.4f}%")
    print(f"   F1 weighted:           {f1_w*100:.4f}%")
    print(f"   F1 macro:              {f1_mac*100:.4f}%")
    print(f"   F1 micro:              {f1_mic*100:.4f}%")
    print(f"   AUC-ROC macro:         {auc_mac:.4f}")
    print(f"   AUC-ROC weighted:      {auc_w:.4f}")
    print(f"   Latencia/muestra:      {t_inf:.4f} ms")
    print(f"   Tiempo entrenamiento:  {t_train:.4f} s")
    print(f"   Parámetros del modelo: {13 * np.prod(tensor_shape):,}")

    # Classification report detallado
    cr = classification_report(y_te, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{'='*65}")
    print("CLASSIFICATION REPORT POR CLASE")
    print(f"{'='*65}")
    print(classification_report(y_te, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    print(f"{'='*65}")
    print("CONFUSION MATRIX (conteos absolutos)")
    print(f"{'='*65}")
    header = "".join([f"{n[:6]:>8}" for n in [c[:6] for c in class_names]])
    print(f"{'':>22}{header}")
    for i, row in enumerate(cm):
        row_str = "".join([f"{v:>8}" for v in row])
        print(f"  {class_names[i][:20]:<20}{row_str}")

    # AUC-ROC por clase
    print(f"\n{'='*65}")
    print("AUC-ROC POR CLASE (One-vs-Rest)")
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

    # Análisis de separabilidad
    print(f"\n{'='*65}")
    print("ANÁLISIS DE SEPARABILIDAD POR CLASE")
    print(f"{'='*65}")
    print(f"  {'Clase':<25} {'Intra':>10} {'Inter':>10} {'Ratio':>8} {'Estado'}")
    print("  " + "-" * 65)
    sep = clf.separability_analysis(X_te, y_te, class_names)
    for name, s in sep.items():
        estado = "✅" if s["ratio"] > 1.2 else ("🟡" if s["ratio"] > 0.9 else "❌")
        print(f"  {name:<25} {s['intra_dist']:>10.3f} {s['inter_dist']:>10.3f} "
              f"{s['ratio']:>8.3f}  {estado}")

    # Distancias entre prototipos
    print(f"\n{'='*65}")
    print("DISTANCIAS ENTRE PROTOTIPOS (par más separado y menos separado)")
    print(f"{'='*65}")
    proto_dist = clf.prototype_distances(class_names)
    np.fill_diagonal(proto_dist, np.inf)
    min_idx = np.unravel_index(np.argmin(proto_dist), proto_dist.shape)
    np.fill_diagonal(proto_dist, 0)
    max_idx = np.unravel_index(np.argmax(proto_dist), proto_dist.shape)
    print(f"   Más separados:  {class_names[max_idx[0]]} vs {class_names[max_idx[1]]}: "
          f"{proto_dist[max_idx]:.4f}")
    print(f"   Menos separados:{class_names[min_idx[0]]} vs {class_names[min_idx[1]]}: "
          f"{proto_dist[min_idx]:.4f}")

    # Análisis específico Espira vs Monofásica-A
    print(f"\n{'='*65}")
    print("ANÁLISIS ESPECÍFICO — ESPIRA-ESPIRA vs MONOFÁSICA-A")
    print(f"{'='*65}")
    mask_bin   = (y_te == 0) | (y_te == 11)
    acc_bin    = float(accuracy_score(y_te[mask_bin], y_pred[mask_bin]))
    proto_dist_esp = float(np.sqrt(np.sum((clf.prototypes_[0] - clf.prototypes_[11])**2)))
    print(f"   Accuracy binario (solo estas 2 clases): {acc_bin*100:.2f}%")
    print(f"   Distancia Frobenius entre prototipos:   {proto_dist_esp:.4f}")
    print(f"   Distancia máxima entre cualquier par:   {np.max(proto_dist):.4f}")
    print(f"   Ratio relativo:                         {proto_dist_esp/np.max(proto_dist):.4f}")

    # Cross-validation 5-fold
    print(f"\n{'='*65}")
    print("CROSS-VALIDATION 5-FOLD (robustez del resultado)")
    print(f"{'='*65}")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_acc = []
    cv_f1  = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        clf_cv = FrobeniusClassifier()
        clf_cv.fit(X[tr_idx], y[tr_idx])
        yp = clf_cv.predict(X[te_idx])
        cv_acc.append(accuracy_score(y[te_idx], yp))
        cv_f1.append(f1_score(y[te_idx], yp, average="weighted"))
        print(f"   Fold {fold+1}: Acc={cv_acc[-1]*100:.2f}%  F1={cv_f1[-1]*100:.2f}%")
    print(f"   {'MEDIA':<8}: Acc={np.mean(cv_acc)*100:.2f}% ± {np.std(cv_acc)*100:.2f}%  "
          f"F1={np.mean(cv_f1)*100:.2f}% ± {np.std(cv_f1)*100:.2f}%")

    # Guardar resultados
    output = {
        "method": "Frobenius Tensorial",
        "timestamp": datetime.now().isoformat(),
        "dataset": {"total": int(len(X)), "train": int(len(X_tr)), "test": int(len(X_te))},
        "model": {
            "params": int(13 * np.prod(tensor_shape)),
            "train_time_s": t_train,
            "inference_ms_per_sample": t_inf,
        },
        "global_metrics": {
            "accuracy": acc, "f1_weighted": f1_w,
            "f1_macro": f1_mac, "f1_micro": f1_mic,
            "auc_roc_macro": auc_mac, "auc_roc_weighted": auc_w,
        },
        "per_class_metrics": cr,
        "auc_roc_per_class": auc_per_class,
        "confusion_matrix": cm.tolist(),
        "separability": sep,
        "binary_espira_monofasica": {
            "accuracy": acc_bin,
            "prototype_distance": proto_dist_esp,
            "max_prototype_distance": float(np.max(proto_dist)),
        },
        "cross_validation": {
            "acc_mean": float(np.mean(cv_acc)),
            "acc_std":  float(np.std(cv_acc)),
            "f1_mean":  float(np.mean(cv_f1)),
            "f1_std":   float(np.std(cv_f1)),
            "folds_acc": [float(v) for v in cv_acc],
            "folds_f1":  [float(v) for v in cv_f1],
        },
    }

    out_path = RESULTS_DIR / "01_frobenius_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 {out_path}")


if __name__ == "__main__":
    main()
