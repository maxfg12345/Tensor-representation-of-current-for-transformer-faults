"""
CONSTRUCCIÓN DEL TENSOR

Se procesa los 36,720 archivos .txt del dataset y genera
el archivo tensor_thec_d.pkl con todos los tensores calculados.

Este archivo es la entrada de los 4 scripts de clasificadores:
  01_method_frobenius.py
  02_method_rf.py
  03_method_cnn.py
  04_method_lstm.py

Tensor resultante por muestra: T ∈ R^(4 × 8 × 8) = 256 valores
  Dimensión 0 (4): componentes de secuencia — I0, I1, I2, I2/I0
  Dimensión 1 (8): features — 4 intra-ventana + 4 inter-ventana
  Dimensión 2 (8): ventanas temporales de medio ciclo (83 muestras)

Dataset: completo sin balancear (36,720 muestras, clases desiguales)
  - 11 clases × 2,160 muestras
  - Espira-espira (tt): 8,640 muestras
  - Devanado-devanado (ww): 4,320 muestras
Split: 80/20 estratificado (definido aquí, respetado por todos los scripts)
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# RUTAS — los 4 scripts de clasificadores leen estas constantes del pkl

DATASET_DIR  = Path("data/ieee_dataset")
OUTPUT_PATH  = Path("data/processed/tensor_thec_d.pkl")

# PARÁMETROS DEL TENSOR — fijos para todos los scripts

FS           = 10000.0          # Hz — del README del dataset
F0           = 60.0             # Hz — sistema de 60 Hz
HALF_CYCLE   = int(FS / (2*F0)) # 83 muestras por ventana de medio ciclo
NUM_WINDOWS  = 8                # ventanas por señal (~4 ciclos completos)
NUM_SEQ      = 4                # I0, I1, I2, ratio I2/I0
NUM_FEATURES = 8                # 4 intra-ventana + 4 inter-ventana
TENSOR_SHAPE = (NUM_SEQ, NUM_FEATURES, NUM_WINDOWS)   # 4×8×8 = 256
RANDOM_STATE = 42
TEST_SIZE    = 0.20             # 80/20 estratificado

# CLASES — orden y nombres fijos para todos los scripts

CLASS_NAMES = [
    "Monofasica-A", "Monofasica-B", "Monofasica-C",
    "Bifasica-AB",  "Bifasica-AC",  "Bifasica-BC",
    "Trifasica-tierra",
    "FaseFase-AB",  "FaseFase-AC",  "FaseFase-BC",
    "Trifasica-sin-tierra",
    "Espira-espira", "Devanado-devanado",
]

FOLDER_TO_CLASS = {
    "class1": 0,  "class2": 1,  "class3": 2,
    "class4": 3,  "class5": 4,  "class6": 5,
    "class7": 6,  "class8": 7,  "class9": 8,
    "class10": 9, "class11": 10,
    "tt": 11,     "ww": 12,
}

# Constantes de Fortescue
A_RE  = np.cos(2 * np.pi / 3)   # -0.5
A_IM  = np.sin(2 * np.pi / 3)   #  0.866
A2_RE = np.cos(4 * np.pi / 3)   # -0.5
A2_IM = np.sin(4 * np.pi / 3)   # -0.866
HARM2_IDX = int(round(2 * F0 * HALF_CYCLE / FS))   # bin FFT de 120 Hz


# FUNCIONES DE CONSTRUCCIÓN DEL TENSOR

def symmetrical_components(ia_w, ib_w, ic_w):
    """Magnitudes |I0|, |I1|, |I2| por Fortescue muestra a muestra."""
    mag_I0 = np.abs((ia_w + ib_w + ic_w) / 3.0)
    I1_re  = (ia_w + A_RE*ib_w + A2_RE*ic_w) / 3.0
    I1_im  = (       A_IM*ib_w + A2_IM*ic_w) / 3.0
    mag_I1 = np.sqrt(I1_re**2 + I1_im**2)
    I2_re  = (ia_w + A2_RE*ib_w + A_RE*ic_w) / 3.0
    I2_im  = (       A2_IM*ib_w + A_IM*ic_w) / 3.0
    mag_I2 = np.sqrt(I2_re**2 + I2_im**2)
    return mag_I0, mag_I1, mag_I2


def energy_2nd_harmonic(signal):
    """Energía normalizada del 2do armónico (120 Hz) en la ventana."""
    n = len(signal)
    if n < 4:
        return 0.0
    spec  = np.abs(np.fft.rfft(signal * np.hanning(n)))**2
    total = np.sum(spec) + 1e-10
    idx   = min(HARM2_IDX, len(spec) - 1)
    return float(spec[idx] / total)


def intra_features(mag):
    """
    4 features intra-ventana calculados dentro de una ventana de 83 muestras.
    Diseñados para ser ortogonales entre sí:
      0: RMS          — energía total de la componente
      1: Entropía     — dispersión de la distribución
      2: E_2arm       — energía del 2do armónico (120 Hz)
      3: Skewness     — asimetría temporal (transitoriedad)
    """
    eps  = 1e-10
    rms  = float(np.sqrt(np.mean(mag**2)))
    hist, _ = np.histogram(mag, bins=8, density=True)
    hist = (hist + eps) / (hist + eps).sum()
    entr = float(-np.sum(hist * np.log2(hist)))
    e2   = energy_2nd_harmonic(mag)
    mean = np.mean(mag)
    std  = np.std(mag) + eps
    skew = float(np.mean(((mag - mean)/std)**3))
    return np.array([rms, entr, e2, skew], dtype=np.float32)


def inter_features(rms_hist, k):
    """
    4 features inter-ventana: capturan la dinámica temporal del
    transitorio a través de las 8 ventanas.
      0: Pendiente    — cambio relativo de RMS respecto a ventana anterior
      1: Aceleración  — cambio de la pendiente (2da derivada)
      2: Relativo     — RMS[k] / RMS[0] (normalizado por ventana inicial)
      3: Curvatura    — 2da diferencia centrada
    """
    if k == 0:
        return np.zeros(4, dtype=np.float32)
    r0   = rms_hist[0]  + 1e-10
    rk   = rms_hist[k]
    rk1  = rms_hist[k-1] + 1e-10
    slope = (rk - rms_hist[k-1]) / rk1
    if k >= 2:
        rk2   = rms_hist[k-2] + 1e-10
        accel = slope - (rms_hist[k-1] - rms_hist[k-2]) / rk2
        curv  = rk - 2*rms_hist[k-1] + rms_hist[k-2]
    else:
        accel = 0.0
        curv  = 0.0
    return np.array([slope, accel, rk/r0, curv], dtype=np.float32)


def build_tensor(ia, ib, ic):
    """
    Construye T ∈ R^(4×8×8) para una señal de falla.
    Proceso:
      1. Dividir en 8 ventanas de HALF_CYCLE muestras
      2. Fortescue por ventana → |I0|, |I1|, |I2|, I2/I0
      3. 4 features intra + 4 features inter por componente
      4. Ensamblar en tensor 4×8×8
    """
    T        = np.zeros(TENSOR_SHAPE, dtype=np.float32)
    rms_hist = {s: [] for s in range(NUM_SEQ)}

    for w in range(NUM_WINDOWS):
        start = w * HALF_CYCLE
        end   = start + HALF_CYCLE
        if end > len(ia):
            break
        ia_w, ib_w, ic_w = ia[start:end], ib[start:end], ic[start:end]
        mag_I0, mag_I1, mag_I2 = symmetrical_components(ia_w, ib_w, ic_w)
        mag_ratio = np.clip(mag_I2 / (mag_I0 + 1e-6), 0, 100)

        for s, comp in enumerate([mag_I0, mag_I1, mag_I2, mag_ratio]):
            intra = intra_features(comp)
            T[s, 0:4, w] = intra
            rms_hist[s].append(float(intra[0]))
            T[s, 4:8, w] = inter_features(rms_hist[s], w)

    return T


# PIPELINE PRINCIPAL

def collect_files():
    """Escanea el dataset y retorna lista de (filepath, class_id)."""
    files   = []
    summary = []
    for folder_name, class_id in FOLDER_TO_CLASS.items():
        folder = DATASET_DIR / folder_name
        if not folder.exists():
            print(f"   ⚠️  No encontrada: {folder}")
            continue
        txts = sorted(folder.glob("*.txt"))
        summary.append((folder_name, class_id, len(txts)))
        for f in txts:
            files.append((f, class_id))

    print(f"\n{'Carpeta':<12} {'Clase':>6} {'Archivos':>10}")
    print("-" * 32)
    for name, cid, n in summary:
        print(f"  {name:<10} {cid:>6} {n:>10,}")
    print(f"\n  Total: {len(files):,} archivos\n")
    return files


def process_all(files):
    """Procesa todos los archivos y construye X, y."""
    n = len(files)
    X = np.zeros((n, *TENSOR_SHAPE), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    errors = 0

    with tqdm(total=n, desc="Construyendo tensores", unit="archivo") as pbar:
        for i, (fp, class_id) in enumerate(files):
            try:
                data = np.loadtxt(fp, delimiter=",")
                ia   = data[:, 1].astype(np.float32)
                ib   = data[:, 2].astype(np.float32)
                ic   = data[:, 3].astype(np.float32)
                X[i] = build_tensor(ia, ib, ic)
                y[i] = class_id
            except Exception:
                errors += 1
            pbar.update(1)

    return X, y, errors


def verify(X, y):
    """Verifica que el output tiene datos reales."""
    all_zero = np.all(X == 0, axis=(1, 2, 3))
    print(f"\n🔍 VERIFICACIÓN:")
    print(f"   Tensores con datos reales: {(~all_zero).sum():,} / {len(X):,}")
    print(f"   Tensores en cero:          {all_zero.sum():,}")
    print(f"\n   Distribución de clases:")
    for cid, name in enumerate(CLASS_NAMES):
        n = (y == cid).sum()
        print(f"     Clase {cid:2d} ({name:<22}): {n:,}")
    return (~all_zero).sum() > 0


def main():
    print("=" * 65)
    print("CONSTRUCCIÓN DEL TENSOR THEC-D")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_PATH}")
    print(f"Tensor:  {TENSOR_SHAPE} = {np.prod(TENSOR_SHAPE)} valores/muestra")
    print(f"FS={FS} Hz | F0={F0} Hz | Ventana={HALF_CYCLE} muestras")
    print("=" * 65)

    if not DATASET_DIR.exists():
        print(f"\n No se encuentra: {DATASET_DIR}")
        return

    # Prueba rápida con 3 archivos antes de procesar todo
    print("\n Prueba rápida con 3 archivos...")
    test_files = []
    for folder_name, class_id in list(FOLDER_TO_CLASS.items())[:3]:
        folder = DATASET_DIR / folder_name
        if folder.exists():
            txts = sorted(folder.glob("*.txt"))
            if txts:
                test_files.append((txts[0], class_id))

    all_ok = True
    for fp, cid in test_files:
        try:
            data   = np.loadtxt(fp, delimiter=",")
            ia, ib, ic = data[:,1].astype(np.float32), data[:,2].astype(np.float32), data[:,3].astype(np.float32)
            tensor = build_tensor(ia, ib, ic)
            nonzero = not np.all(tensor == 0)
            status  = "✅" if nonzero else "❌ TODO CERO"
            print(f"   {status}  {fp.name}  I0_RMS_w0={tensor[0,0,0]:.4f}  I2_RMS_w0={tensor[2,0,0]:.4f}")
            if not nonzero:
                all_ok = False
        except Exception as e:
            print(f"   ❌ Error: {fp.name} — {e}")
            all_ok = False

    if not all_ok:
        print("\n❌ Prueba rápida fallida. Revisar archivos antes de continuar.")
        return

    print("\n✅ Prueba rápida exitosa.")
    respuesta = input("\n¿Procesar los 36,720 archivos completos? (s/n): ").strip().lower()
    if respuesta != "s":
        print("Cancelado.")
        return

    # Recolectar y procesar
    files = collect_files()
    t_start = datetime.now()
    X, y, errors = process_all(files)
    t_elapsed = (datetime.now() - t_start).total_seconds() / 60

    print(f"\n⏱️  Tiempo: {t_elapsed:.1f} minutos")
    print(f"   Procesados: {len(files) - errors:,} | Errores: {errors}")

    # Verificar
    ok = verify(X, y)
    if not ok:
        print("\n❌ ABORTANDO — tensores vacíos detectados.")
        return

    # Guardar
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        # Datos
        "X": X,
        "y": y,

        # Metadatos — los 4 scripts los leen directamente de aquí
        "tensor_shape":    list(TENSOR_SHAPE),
        "class_names":     CLASS_NAMES,
        "folder_to_class": FOLDER_TO_CLASS,
        "num_classes":     len(CLASS_NAMES),
        "random_state":    RANDOM_STATE,
        "test_size":       TEST_SIZE,

        # Parámetros del tensor
        "fs":          FS,
        "f0":          F0,
        "half_cycle":  HALF_CYCLE,
        "num_windows": NUM_WINDOWS,
        "num_seq":     NUM_SEQ,
        "num_features": NUM_FEATURES,

        # Descripción de features
        "seq_names":   ["I0", "I1", "I2", "I2_over_I0"],
        "feature_names": [
            "RMS", "Entropy", "E_2harm", "Skewness",
            "Slope", "Acceleration", "Relative", "Curvature"
        ],
        "feature_types": {
            "intra_window": [0, 1, 2, 3],
            "inter_window": [4, 5, 6, 7],
        },

        # Info del dataset
        "dataset_info": {
            "source": "Bera et al. IEEE DataPort — Transients and Faults in Power Transformers and PARs",
            "simulation": "PSCAD/EMTDC",
            "system": "5-bus, 500MVA, 500kV/230kV, 60Hz",
            "signal_type": "3-phase differential currents",
            "samples_per_signal": 726,
            "window_duration_s": 0.0725,
            "fault_inception": "15.0s",
            "fault_duration": "0.05s (3 cycles)",
            "total_samples": int(len(X)),
            "errors": int(errors),
        },

        "build_timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)

    size_mb = OUTPUT_PATH.stat().st_size / (1024**2)
    print(f"\n💾 Guardado: {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    print(f"""
{'='*65}
✅ TENSOR THEC-D CONSTRUIDO EXITOSAMENTE

  Archivo:       {OUTPUT_PATH}
  Muestras:      {len(X):,}
  Shape tensor:  {TENSOR_SHAPE}
  Features:      {np.prod(TENSOR_SHAPE)} por muestra
  Split train/test: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} estratificado
  Random state:  {RANDOM_STATE}

Siguientes pasos (en cualquier orden):
  python scripts/01_method_frobenius.py
  python scripts/02_method_rf.py
  python scripts/03_method_cnn.py
  python scripts/04_method_lstm.py
{'='*65}
""")


if __name__ == "__main__":
    main()
