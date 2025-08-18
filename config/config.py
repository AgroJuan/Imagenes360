import os
import multiprocessing

# %% CONFIGURACIÓN
DATASET_DIR = r"J:\Imagenes360\Train"
TEST_DIRS = [r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Sabanalarga\2. Fotografias 360"]

# TEST_DIRS = [
#     r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Dabeiba\2. Fotografias 360",
#     r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Nechi\2. Fotografias 360",
#     r"J:\Imagenes360\Maceo\2. Fotografias 360",r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Nechi\2. Fotografias 360",
#     r"D:\Imagenes360\Puerto Berrio\2. Fotografias 360",
#     r"D:\Imagenes360\Sonson\2. Fotografias 360"
# ]

CLASES = {'borrosa': 0, 'aceptable': 1, 'nitida': 2}
CLASES_INV = {v: k for k, v in CLASES.items()}
FACE_NAMES = ['Frontal', 'Derecha', 'Trasera', 'Izquierda']
MODELO_PATH = os.path.join(DATASET_DIR, 'modelo_xgb.pkl')
SCALER_PATH = os.path.join(DATASET_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(DATASET_DIR, 'features.pkl')
N_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))
THRESHOLD_HIGH_ENTROPY = 0.7  # 80% de entropía
THRESHOLD_MED_ENTROPY = 0.6  # 60% de entropía
THRESHOLD_HIGH_BLUR = 0.60  # 60% de probabilidad de borroso
THRESHOLD_MED_BLUR = 0.30   # 40% de probabilidad de borroso
DIFF_PROB = 0.2            # Diferencia significativa
OPTIMIZAR_HIPERPARAMETROS = True
ENTRENAR_NUEVO_MODELO = False