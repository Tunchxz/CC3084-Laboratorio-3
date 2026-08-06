"""Preprocesamiento del dataset ASL Alphabet: submuestra, reescalado y particion.

Convierte las ~87,000 fotografias de 200x200 que hay en data/raw/ en arreglos
de NumPy listos para entrenar, guardados en data/processed/.

Decisiones de diseno (justificadas en notebooks/02_preprocesamiento.ipynb):

* **Submuestra de 500 imagenes por clase.** El dataset completo no cabe en la
  memoria de la maquina de trabajo (2.8 GiB de RAM total). Con 500 por clase se
  conserva el balance perfecto entre las 29 clases y el arreglo resultante pesa
  ~178 MB.
* **Resolucion 64x64.** A ese tamano la forma de la mano sigue siendo legible y
  el entrenamiento en CPU es viable. Bajar a 32x32 borra los detalles que
  distinguen letras parecidas (M/N/S).
* **Recorte de 3 pixeles en cada borde.** Las fotografias traen un marco azul
  saturado grabado en los 3 pixeles del margen (un artefacto de la captura). No
  aporta informacion de clase, se convierte en bandas azules al aplicar aumento
  geometrico y no existira en fotos tomadas con otra camara.
* **Almacenamiento en uint8.** Guardar en float32 cuadruplicaria la memoria. La
  normalizacion a [0, 1] se aplica por lote durante el entrenamiento (con una
  capa Rescaling), no sobre el arreglo completo.
* **Particion 70/15/15 estratificada** con semilla fija, porque el conjunto de
  prueba oficial de Kaggle solo trae 28 imagenes (una por clase) y no permite
  medir nada de forma confiable.

Uso
---
    python src/preprocess.py                       # 500 por clase, 64x64
    python src/preprocess.py --per-class 800 --size 96
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLASES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]

PER_CLASS = 500     # imagenes por clase en la submuestra
SIZE = 64           # lado del cuadrado al que se reescala cada imagen
SEED = 42           # semilla unica para submuestra y particiones
RECORTE = 3         # pixeles que se recortan de cada borde (marco azul de captura)


# --------------------------------------------------------------------------
# Submuestra
# --------------------------------------------------------------------------
def submuestrear(raw_dir=RAW_DIR, per_class=PER_CLASS, seed=SEED):
    """Elige `per_class` archivos al azar de cada clase, de forma determinista.

    Se muestrea al azar (y no los primeros N) porque las imagenes de cada clase
    provienen de secuencias de video: tomar los primeros N archivos daria un
    bloque de fotogramas casi identicos entre si.

    Devuelve (rutas, etiquetas) donde etiquetas son indices enteros de CLASES.
    """
    rng = random.Random(seed)
    rutas, etiquetas = [], []
    for idx, clase in enumerate(CLASES):
        disponibles = sorted((Path(raw_dir) / clase).glob("*.jpg"))
        if len(disponibles) < per_class:
            msg = f"La clase {clase} solo tiene {len(disponibles)} imagenes (<{per_class})"
            raise RuntimeError(msg)
        elegidas = rng.sample(disponibles, per_class)
        rutas.extend(sorted(elegidas))
        etiquetas.extend([idx] * per_class)
    return rutas, np.array(etiquetas, dtype="int64")


# --------------------------------------------------------------------------
# Carga y reescalado
# --------------------------------------------------------------------------
def cargar_imagen(ruta, size=SIZE, recorte=RECORTE):
    """Abre una imagen, recorta el borde, la pasa a RGB y la reescala (uint8).

    El recorte quita `recorte` pixeles de cada lado antes de reescalar, para
    eliminar el marco azul que las fotografias traen grabado en el margen.
    """
    with Image.open(ruta) as img:
        img = img.convert("RGB")
        if recorte:
            ancho, alto = img.size
            img = img.crop((recorte, recorte, ancho - recorte, alto - recorte))
        return np.asarray(img.resize((size, size), Image.BILINEAR), dtype="uint8")


def construir_arreglo(rutas, size=SIZE, recorte=RECORTE, verbose=True):
    """Convierte una lista de rutas en un tensor uint8 (n, size, size, 3).

    El arreglo se reserva de una sola vez y se llena en el sitio para no
    acumular una lista intermedia de n imagenes en memoria.
    """
    X = np.empty((len(rutas), size, size, 3), dtype="uint8")
    for i, ruta in enumerate(rutas):
        X[i] = cargar_imagen(ruta, size, recorte)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{len(rutas):,} imagenes procesadas")
    return X


# --------------------------------------------------------------------------
# Particion
# --------------------------------------------------------------------------
def dividir(X, y, seed=SEED):
    """Particion estratificada 70 % entrenamiento / 15 % validacion / 15 % prueba.

    Se hace en dos pasos: primero se separa el 30 % de reserva y luego ese 30 %
    se parte a la mitad. `stratify` mantiene las 29 clases balanceadas en las
    tres particiones.
    """
    X_train, X_resto, y_train, y_resto = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_resto, y_resto, test_size=0.50, random_state=seed, stratify=y_resto
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# --------------------------------------------------------------------------
# Guardado
# --------------------------------------------------------------------------
def guardar(particiones, out_dir=PROCESSED_DIR, per_class=PER_CLASS, size=SIZE,
            seed=SEED, recorte=RECORTE):
    """Escribe los .npy y los metadatos en data/processed/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = particiones
    datos = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for nombre, (X, y) in datos.items():
        np.save(out_dir / f"X_{nombre}.npy", X)
        np.save(out_dir / f"y_{nombre}.npy", y)

    (out_dir / "clases.json").write_text(json.dumps(CLASES, indent=2))
    metadata = {
        "per_class": per_class,
        "size": size,
        "seed": seed,
        "recorte_borde_px": recorte,
        "dtype": "uint8",
        "canales": "RGB",
        "normalizacion": "por lote, con keras.layers.Rescaling(1/255) en el modelo",
        "split": {"train": 0.70, "val": 0.15, "test": 0.15, "estratificado": True},
        "conteos": {nombre: int(len(y)) for nombre, (_, y) in datos.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def cargar_procesado(out_dir=PROCESSED_DIR, mmap=True):
    """Lee los .npy generados. Con mmap=True no carga todo a RAM de golpe."""
    out_dir = Path(out_dir)
    modo = "r" if mmap else None
    datos = {}
    for nombre in ("train", "val", "test"):
        datos[nombre] = (np.load(out_dir / f"X_{nombre}.npy", mmap_mode=modo),
                         np.load(out_dir / f"y_{nombre}.npy"))
    clases = json.loads((out_dir / "clases.json").read_text())
    return datos, clases


# --------------------------------------------------------------------------
# Pipeline completo
# --------------------------------------------------------------------------
def ejecutar(raw_dir=RAW_DIR, out_dir=PROCESSED_DIR, per_class=PER_CLASS,
             size=SIZE, seed=SEED, recorte=RECORTE, verbose=True):
    """Corre el preprocesamiento completo y devuelve los metadatos."""
    if verbose:
        print(f"Submuestreando {per_class} imagenes de cada una de las {len(CLASES)} clases...")
    rutas, y = submuestrear(raw_dir, per_class, seed)

    if verbose:
        mb = len(rutas) * size * size * 3 / 1e6
        print(f"Recortando {recorte} px de cada borde y cargando {len(rutas):,} imagenes "
              f"a {size}x{size} (~{mb:.0f} MB en uint8)...")
    X = construir_arreglo(rutas, size, recorte, verbose)

    if verbose:
        print("Particionando 70/15/15 de forma estratificada...")
    particiones = dividir(X, y, seed)

    metadata = guardar(particiones, out_dir, per_class, size, seed, recorte)
    if verbose:
        print(f"Guardado en {out_dir}: {metadata['conteos']}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocesa el dataset ASL Alphabet")
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--recorte", type=int, default=RECORTE,
                        help="pixeles a recortar de cada borde (default: 3)")
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    ejecutar(args.raw, args.out, args.per_class, args.size, args.seed, args.recorte)


if __name__ == "__main__":
    main()
