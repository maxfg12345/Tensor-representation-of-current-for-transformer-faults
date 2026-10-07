"""
04_method_lstm.py  —  MÉTODO 4: LSTM SOBRE TENSOR THEC-D

Lee: data/processed/tensor_thec_d.pkl  (generado por 00_build_tensor_pkl.py)

LSTM que procesa el tensor THEC-D como secuencia temporal de 8 pasos.
Cada paso es un vector de 4×8 = 32 features (4 componentes de secuencia
× 8 features por componente), representando la firma espectral en una
ventana de medio ciclo.

La dimensión temporal del tensor (8 ventanas) es la secuencia natural
que el LSTM procesa — captura dependencias entre ventanas que la CNN-2D
y el RF no modelan explícitamente.

Arquitectura:
  Input: (8, 32) — 8 timesteps × 32 features por timestep
  LSTM(64, return_sequences=True)
  Dropout(0.2)
  LSTM(32)
  Dropout(0.2)
  Dense(64) + Dropout
  Dense(13, softmax)

Resultados guardados en: results/04_lstm_results.json
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
    X_raw = d["X"]   # (N, 4, 8, 8) — (N, seq, features, windows)

    # Reordenar para LSTM: (N, timesteps=8, features_per_step=4×8=32)
    # Transponer de (N, 4, 8, 8) a (N, 8, 4, 8) y luego aplanar últimas 2 dims
    X = np.transpose(X_raw, (0, 3, 1, 2))   # (N, 8_windows, 4_seq, 8_feats)
    X = X.reshape(X.shape[0], X.shape[1], -1).astype(np.float32)  # (N, 8, 32)

    y = d["y"]
    print(f"   X (timesteps, features): {X.shape}  |  y: {y.shape}")
    print(f"   Interpretación: 8 ventanas temporales × 32 features/ventana")
    print(f"   Cada timestep = firma espectral de un medio ciclo de 60Hz")
    return X, y, d["class_names"], d["random_state"], d["test_size"]


def build_model(input_shape, num_classes):
    """
    LSTM apilado que procesa la secuencia temporal de 8 ventanas.
    Cada ventana es un vector de 32 features (4 componentes × 8 features).

    La clave: el LSTM puede capturar que en espira-espira, la secuencia
    I2/I0 en ventanas 1→8 tiene un patrón temporal diferente al de
    Monofásica-A — algo que el RF y la CNN-2D no modelan explícitamente.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),

        # Primera capa LSTM — return_sequences=True para apilar
        keras.layers.LSTM(64, return_sequences=True,
                          dropout=0.2, recurrent_dropout=0.1),

        # Segunda capa LSTM — procesa la salida de la primera
        keras.layers.LSTM(32, return_sequences=False,
                          dropout=0.2, recurrent_dropout=0.1),

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
    print("MÉTODO 4 — LSTM SOBRE SECUENCIA TEMPORAL DEL TENSOR THEC-D")
    print(f"   TensorFlow: {tf.__version__}  |  Keras: {keras.__version__}")
    print("=" * 65 + "\n")

    X, y, class_names, random_state, test_size = load_data()
    num_classes = len(class_names)
    input_shape = X.shape[1:]   # (8, 32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"   Train: {len(X_tr):,}  |  Test: {len(X_te):,}")
    print(f"   Input shape: {input_shape}  (timesteps=8, features=32)\n")

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
            filepath=str(RESULTS_DIR / "04_lstm_best.keras"),
            monitor="val_accuracy", save_best_only=True),
    ]

    print("\n🚀 Entrenando LSTM...")
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
    print("RESULTADOS GLOBALES — LSTM")
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

    print(f"\n{'='*65}")
    print("CURVAS DE ENTRENAMIENTO (resumen)")
    print(f"{'='*65}")
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch   = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"   Mejor val_accuracy: {best_val_acc*100:.2f}% (época {best_epoch})")
    print(f"   Train accuracy final: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"   Val accuracy final:   {history.history['val_accuracy'][-1]*100:.2f}%")

    # Análisis temporal: ¿el LSTM aprovecha la secuencia?
    print(f"\n{'='*65}")
    print("ANÁLISIS DE DEPENDENCIAS TEMPORALES")
    print(f"{'='*65}")
    print(f"   Si el LSTM supera al RF, significa que la dependencia")
    print(f"   entre ventanas temporales aporta información adicional")
    print(f"   más allá de los features inter-ventana del tensor.")
    print(f"   LSTM Accuracy: {acc*100:.2f}%")
    print(f"   (Comparar contra RF en 02_rf_results.json)")

    diff = abs(acc - f1_w)
    print(f"\n{'='*65}")
    print("VERIFICACIÓN DE CONSISTENCIA")
    print(f"{'='*65}")
    print(f"   Accuracy:    {acc*100:.4f}%")
    print(f"   F1 weighted: {f1_w*100:.4f}%")
    print(f"   Diferencia:  {diff*100:.4f} puntos")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    output = {
        "method": "LSTM sobre secuencia temporal del Tensor THEC-D",
        "timestamp": datetime.now().isoformat(),
        "architecture": "Input(8,32)->LSTM(64,seq=True)->Drop->LSTM(32)->Drop->Dense(64)->Dense(13)",
        "input_interpretation": "8 ventanas de medio ciclo × 32 features (4 componentes × 8 features)",
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
    out_path = RESULTS_DIR / "04_lstm_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 {out_path}")


if __name__ == "__main__":
    main()
