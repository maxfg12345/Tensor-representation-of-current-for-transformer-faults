"""
03_method_cnn.py  —  MÉTODO 3: CNN-2D SOBRE TENSOR THEC-D

Lee: data/processed/tensor_thec_d.pkl  (generado por 00_build_tensor_pkl.py)

CNN-2D que recibe el tensor (4, 8, 8) directamente como entrada.
Opera sobre las dimensiones features×ventanas para cada componente de
secuencia, aprendiendo patrones locales en el espacio 2D.

Arquitectura:
  Input: (4, 8, 8) — tratado como imagen 8×8 con 4 canales
  Conv2D(32, 3×3) + BN + Dropout
  Conv2D(64, 3×3) + BN + Dropout
  GlobalAveragePooling2D
  Dense(64) + Dropout
  Dense(13, softmax)

Resultados guardados en: results/03_cnn_results.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

PKL_PATH    = Path("data/processed/tensor_thec_d.pkl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS     = 150
BATCH_SIZE = 64


def load_data():
    print(f"📥 Cargando: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        d = pickle.load(f)
    # CNN-2D: tensor (N, 4, 8, 8) → reordenar a (N, 8, 8, 4) para Keras
    # (height=8, width=8, channels=4)
    X = np.transpose(d["X"], (0, 2, 3, 1)).astype(np.float32)
    y = d["y"]
    print(f"   X (H×W×C): {X.shape}  |  y: {y.shape}")
    return X, y, d["class_names"], d["random_state"], d["test_size"]


def build_model(input_shape, num_classes):
    """
    CNN-2D sobre el tensor THEC-D.
    Input: (8, 8, 4) — 8 features × 8 ventanas × 4 componentes de secuencia.
    La convoluci\u00f3n 2D opera sobre el espacio features\u00d7ventanas preservando
    la relaci\u00f3n espacial entre dimensiones del tensor.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.GlobalAveragePooling2D(),

        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.L2(1e-4)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 65)
    print("MÉTODO 3 — CNN-2D SOBRE TENSOR THEC-D")
    print(f"   TensorFlow: {tf.__version__}  |  Keras: {keras.__version__}")
    print("=" * 65 + "\n")

    X, y, class_names, random_state, test_size = load_data()
    num_classes  = len(class_names)
    input_shape  = X.shape[1:]   # (8, 8, 4)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"   Train: {len(X_tr):,}  |  Test: {len(X_te):,}")
    print(f"   Input shape: {input_shape}\n")

    model = build_model(input_shape, num_classes)
    model.summary()
    total_params = model.count_params()
    print(f"\n🧠 Parámetros totales: {total_params:,}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=7, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            filepath=str(RESULTS_DIR / "03_cnn_best.keras"),
            monitor="val_accuracy", save_best_only=True),
    ]

    print("\n🚀 Entrenando CNN-2D...")
    t0 = datetime.now()
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    t_train = (datetime.now() - t0).total_seconds()
    epochs_run = len(history.history["loss"])
    print(f"\n⏱️  {t_train/60:.1f} minutos  |  {epochs_run} épocas")

    t0      = datetime.now()
    y_proba = model.predict(X_te, verbose=0)
    t_inf   = (datetime.now() - t0).total_seconds() * 1000 / len(X_te)
    y_pred  = y_proba.argmax(axis=1)

    acc    = float(accuracy_score(y_te, y_pred))
    f1_w   = float(f1_score(y_te, y_pred, average="weighted"))
    f1_mac = float(f1_score(y_te, y_pred, average="macro"))
    f1_mic = float(f1_score(y_te, y_pred, average="micro"))
    y_bin  = label_binarize(y_te, classes=list(range(num_classes)))
    try:
        auc_mac = float(roc_auc_score(y_bin, y_proba, average="macro",    multi_class="ovr"))
        auc_w   = float(roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovr"))
    except Exception:
        auc_mac = auc_w = 0.0

    print(f"\n{'='*65}")
    print("RESULTADOS GLOBALES — CNN-2D")
    print(f"{'='*65}")
    print(f"   Accuracy:              {acc*100:.4f}%")
    print(f"   F1 weighted:           {f1_w*100:.4f}%")
    print(f"   F1 macro:              {f1_mac*100:.4f}%")
    print(f"   F1 micro:              {f1_mic*100:.4f}%")
    print(f"   AUC-ROC macro:         {auc_mac:.4f}")
    print(f"   AUC-ROC weighted:      {auc_w:.4f}")
    print(f"   Latencia/muestra:      {t_inf:.4f} ms")
    print(f"   Tiempo entrenamiento:  {t_train:.4f} s")
    print(f"   Épocas ejecutadas:     {epochs_run}")
    print(f"   Parámetros del modelo: {total_params:,}")

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

    print(f"\n{'='*65}")
    print("ANÁLISIS ESPECÍFICO — ESPIRA-ESPIRA vs MONOFÁSICA-A")
    print(f"{'='*65}")
    mask_bin = (y_te == 0) | (y_te == 11)
    acc_bin  = float(accuracy_score(y_te[mask_bin], y_pred[mask_bin]))
    print(f"   Accuracy binario: {acc_bin*100:.2f}%")

    # Curvas de entrenamiento
    print(f"\n{'='*65}")
    print("CURVAS DE ENTRENAMIENTO (resumen)")
    print(f"{'='*65}")
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch   = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"   Mejor val_accuracy: {best_val_acc*100:.2f}% (época {best_epoch})")
    print(f"   Train accuracy final: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"   Val accuracy final:   {history.history['val_accuracy'][-1]*100:.2f}%")

    # Consistencia acc vs f1
    diff = abs(acc - f1_w)
    print(f"\n{'='*65}")
    print("VERIFICACIÓN DE CONSISTENCIA")
    print(f"{'='*65}")
    print(f"   Accuracy:   {acc*100:.4f}%")
    print(f"   F1 weighted:{f1_w*100:.4f}%")
    print(f"   Diferencia: {diff*100:.4f} puntos")
    if diff > 0.01:
        print(f"   ⚠️  Diferencia > 1 punto — dataset desbalanceado afecta la métrica")
    else:
        print(f"   ✅ Consistente")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    output = {
        "method": "CNN-2D sobre Tensor THEC-D",
        "timestamp": datetime.now().isoformat(),
        "architecture": "Input(8,8,4)->Conv2D(32)->BN->Drop->Conv2D(64)->BN->Drop->GAP->Dense(64)->Dense(13)",
        "dataset": {"total": int(len(X)), "train": int(len(X_tr)), "test": int(len(X_te))},
        "model": {
            "params": int(total_params),
            "epochs_run": epochs_run,
            "train_time_s": t_train,
            "inference_ms_per_sample": t_inf,
        },
        "global_metrics": {"accuracy": acc, "f1_weighted": f1_w,
                           "f1_macro": f1_mac, "f1_micro": f1_mic,
                           "auc_roc_macro": auc_mac, "auc_roc_weighted": auc_w},
        "per_class_metrics": cr,
        "auc_roc_per_class": auc_per_class,
        "confusion_matrix": cm.tolist(),
        "binary_espira_monofasica": {"accuracy": acc_bin},
        "training_history": history_dict,
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
    }
    out_path = RESULTS_DIR / "03_cnn_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 {out_path}")


if __name__ == "__main__":
    main()
