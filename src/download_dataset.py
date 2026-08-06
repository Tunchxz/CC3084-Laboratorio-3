"""Descarga el dataset ASL Alphabet de Kaggle y lo deja listo en data/raw/.

El dataset (https://www.kaggle.com/datasets/grassknoted/aslalphabet) trae unas
87,000 fotografias .jpg de 200x200 a color, organizadas en 29 carpetas: las 26
letras del alfabeto mas SPACE, DELETE y NOTHING.

Autenticacion
-------------
`kagglehub` lee las credenciales automaticamente de una de estas fuentes (en
orden): la variable KAGGLE_API_TOKEN, el archivo ~/.kaggle/access_token o el
archivo legado ~/.kaggle/kaggle.json. Basta con tener una configurada; el token
se genera en Kaggle -> Settings -> API.

Uso
---
    python src/download_dataset.py            # descarga (o usa la cache) y enlaza a data/raw
    python src/download_dataset.py --info     # solo reporta que hay en disco

Desde un cuaderno
-----------------
    import download_dataset as dl
    conteos = dl.ensure_dataset()      # idempotente: si ya esta, no descarga
    inv = dl.inventario()              # DataFrame con resolucion/formato por archivo
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

# Carpetas del proyecto: este script vive en <proyecto>/src/
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_TEST_DIR = BASE_DIR / "data" / "raw_test_oficial"

DATASET_ID = "grassknoted/asl-alphabet"

# Las 29 clases del dataset, en el orden alfabetico que usa Kaggle.
CLASES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]

# Cuantas imagenes trae cada clase en el conjunto de entrenamiento oficial.
IMAGENES_POR_CLASE_ESPERADAS = 3000


# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------
def descargar_kaggle():
    """Descarga el dataset con kagglehub y devuelve la ruta de la cache local.

    kagglehub guarda en ~/.cache/kagglehub/ y no vuelve a descargar si ya
    existe, asi que llamar esta funcion varias veces es seguro.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        msg = "Falta kagglehub. Instalar con: pip install -r requirements.txt"
        raise RuntimeError(msg) from exc

    try:
        ruta = kagglehub.dataset_download(DATASET_ID)
    except Exception as exc:
        msg = (
            f"No se pudo descargar {DATASET_ID} de Kaggle: {exc}\n"
            "Revise sus credenciales. Genere un token en Kaggle -> Settings -> API\n"
            "y guardelo en ~/.kaggle/access_token (o exporte KAGGLE_API_TOKEN)."
        )
        raise RuntimeError(msg) from exc

    return Path(ruta)


def _buscar_carpeta_clases(raiz):
    """Ubica la carpeta que contiene las 29 subcarpetas de clases.

    El zip de Kaggle anida las carpetas de forma redundante
    (asl_alphabet_train/asl_alphabet_train/A/...), asi que en vez de asumir la
    ruta se busca la primera carpeta que tenga subcarpetas con nombre de clase.
    """
    clases_set = set(CLASES)
    for carpeta in [raiz, *sorted(p for p in raiz.rglob("*") if p.is_dir())]:
        subdirs = {p.name for p in carpeta.iterdir() if p.is_dir()}
        # Se pide una coincidencia amplia para tolerar variaciones de nombres.
        if len(subdirs & clases_set) >= 26:
            return carpeta
    msg = f"No se encontro la carpeta con las clases dentro de {raiz}"
    raise RuntimeError(msg)


def _enlazar(origen, destino):
    """Crea un symlink destino -> origen (o copia si el symlink no es posible).

    Se usa un enlace simbolico para no duplicar ~1 GB en disco: la cache de
    kagglehub queda como unica copia real de los datos crudos.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.is_symlink() or destino.exists():
        return
    try:
        destino.symlink_to(origen, target_is_directory=origen.is_dir())
    except OSError:
        shutil.copytree(origen, destino)


def ensure_dataset(out_dir=RAW_DIR, test_dir=RAW_TEST_DIR):
    """Garantiza que data/raw tenga las 29 carpetas de clases.

    Si ya estan, no descarga nada. Devuelve un diccionario clase -> numero de
    imagenes .jpg encontradas.
    """
    out_dir = Path(out_dir)
    faltantes = [c for c in CLASES if not (out_dir / c).is_dir()]

    if faltantes:
        print(f"Faltan {len(faltantes)} clases en {out_dir}; descargando de Kaggle...")
        cache = descargar_kaggle()
        print(f"  cache de kagglehub: {cache}")

        origen_train = _buscar_carpeta_clases(cache)
        print(f"  carpeta de clases: {origen_train}")
        for clase in CLASES:
            _enlazar(origen_train / clase, out_dir / clase)

        # El test oficial de Kaggle son archivos sueltos (A_test.jpg, ...).
        pruebas = [p for p in cache.rglob("*_test.jpg")]
        if pruebas:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            for p in pruebas:
                _enlazar(p, Path(test_dir) / p.name)
            print(f"  test oficial: {len(pruebas)} imagenes en {test_dir}")
    else:
        print(f"Las 29 clases ya estan en {out_dir}; se omite la descarga.")

    return contar_imagenes(out_dir)


# --------------------------------------------------------------------------
# Inspeccion
# --------------------------------------------------------------------------
def contar_imagenes(out_dir=RAW_DIR):
    """Cuenta los .jpg de cada clase sin abrir las imagenes."""
    out_dir = Path(out_dir)
    return {c: len(list((out_dir / c).glob("*.jpg"))) if (out_dir / c).is_dir() else 0
            for c in CLASES}


def inventario(out_dir=RAW_DIR, por_clase=None):
    """DataFrame con los metadatos de cada imagen: clase, tamano, modo y bytes.

    Solo se lee el **encabezado** de cada archivo (`Image.open` es perezoso: no
    decodifica los pixeles hasta que se accede a ellos). Asi se pueden recorrer
    los 87,000 archivos usando muy poca memoria.

    `por_clase` limita cuantos archivos se revisan por clase (util para un
    muestreo rapido); None revisa todos.
    """
    out_dir = Path(out_dir)
    filas = []
    for clase in CLASES:
        carpeta = out_dir / clase
        if not carpeta.is_dir():
            continue
        archivos = sorted(carpeta.glob("*.jpg"))
        if por_clase is not None:
            archivos = archivos[:por_clase]
        for p in archivos:
            with Image.open(p) as img:      # perezoso: no decodifica pixeles
                ancho, alto = img.size
                modo, formato = img.mode, img.format
            filas.append({"clase": clase, "archivo": p.name,
                          "ancho": ancho, "alto": alto,
                          "modo": modo, "formato": formato,
                          "bytes": os.path.getsize(p)})
    return pd.DataFrame(filas)


def rutas_por_clase(out_dir=RAW_DIR):
    """Diccionario clase -> lista ordenada de rutas .jpg."""
    out_dir = Path(out_dir)
    return {c: sorted((out_dir / c).glob("*.jpg")) for c in CLASES
            if (out_dir / c).is_dir()}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset ASL Alphabet")
    parser.add_argument("--out", type=Path, default=RAW_DIR,
                        help="carpeta de salida (default: data/raw)")
    parser.add_argument("--info", action="store_true",
                        help="solo reportar lo que ya hay en disco")
    args = parser.parse_args()

    conteos = contar_imagenes(args.out) if args.info else ensure_dataset(args.out)

    total = sum(conteos.values())
    print(f"\nClases encontradas: {sum(1 for v in conteos.values() if v > 0)}/29")
    print(f"Total de imagenes:  {total:,}")
    desbalance = {c: n for c, n in conteos.items() if n != IMAGENES_POR_CLASE_ESPERADAS}
    if desbalance:
        print(f"Clases con != {IMAGENES_POR_CLASE_ESPERADAS} imagenes: {desbalance}")
    else:
        print(f"Todas las clases tienen {IMAGENES_POR_CLASE_ESPERADAS} imagenes.")


if __name__ == "__main__":
    main()
