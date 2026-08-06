# Laboratorio 3 — Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)

Clasificador de letras del alfabeto de Lenguaje de Señas Americano (ASL) a partir de fotografías de manos, para el caso de **SignBridge**. Se usa el dataset [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/aslalphabet) de Kaggle: ~87,000 imágenes JPG de 200×200 a color, repartidas en 29 clases (A–Z más `space`, `del` y `nothing`).

## Requisitos previos

### 1. Python

El proyecto corre sobre **Python 3.14**.

### 2. Credenciales de Kaggle

El dataset requiere autenticación. En Kaggle: **Settings → API → Create New Token**, y luego guardar el token en `~/.kaggle/access_token`. `kagglehub` lo lee de ahí automáticamente. También sirve el archivo legado `~/.kaggle/kaggle.json` o la variable `KAGGLE_API_TOKEN`.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name lab3-asl --display-name "Lab3 ASL (.venv)"
```

## Uso

```bash
# 1. Descargar el dataset (~1 GB; solo la primera vez)
.venv/bin/python src/download_dataset.py

# 2. Generar los arreglos de entrenamiento (500 imagenes por clase a 64x64)
.venv/bin/python src/preprocess.py

# 3. Abrir los cuadernos
.venv/bin/jupyter lab
```

Los cuadernos se ejecutan en orden y con el kernel **Lab3 ASL (.venv)**.

## Estructura del proyecto

```
CC3084-Laboratorio-3/
├── src/
│   ├── download_dataset.py        # descarga de Kaggle + inventario de metadatos
│   └── preprocess.py              # submuestra, reescalado, particion y guardado
├── notebooks/
│   ├── 01_eda.ipynb               # analisis exploratorio (ejercicios 1 y 2)
│   ├── 02_preprocesamiento.ipynb  # preprocesamiento (ejercicio 3)
│   └── 03_seleccion_modelos.ipynb # seleccion de modelos y plan de procesamiento
├── data/raw/                      # datos crudos de Kaggle (no se versiona)
├── data/processed/                # arreglos .npy generados (no se versiona)
├── requirements.txt
└── README.md
```
