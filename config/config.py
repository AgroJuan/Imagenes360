import os
import multiprocessing

# %% CONFIGURACIÓN

DIR = r"J:\1_Trabajo\Conestudios SAS\0_Proyectos\0_Activo\VALOR+ Imagenes360\Imagenes360"
MODEL_DIR = os.path.join(DIR, 'models')
TRAIN_DIR = os.path.join(DIR, 'train')

TEST_DIRS = [r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Sabanalarga\2. Fotografias 360"]

# TEST_DIRS = [
#     r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Dabeiba\2. Fotografias 360",
#     r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Nechi\2. Fotografias 360",
#     r"J:\Imagenes360\Maceo\2. Fotografias 360",r"C:\Users\Juan Camilo\Downloads\Imagenes 360\Nechi\2. Fotografias 360",
#     r"D:\Imagenes360\Puerto Berrio\2. Fotografias 360",
#     r"D:\Imagenes360\Sonson\2. Fotografias 360"
# ]

CLASES_DESENFOQUE = {'borrosa': 0, 'aceptable': 1, 'nitida': 2}
CLASES_EXPOSICION = {'subexpuesta': 0, 'correcta': 1, 'sobrexpuesta': 0}

PESO_DESENFOQUE = 0.6      # 60% importancia al desenfoque
PESO_EXPOSICION = 0.4      # 40% importancia a la exposición

FACE_NAMES = ['Frontal', 'Derecha', 'Trasera', 'Izquierda']

MODELO_EXPOSICION_PATH = os.path.join(MODEL_DIR, 'modelo_exposicion.pkl')
SCALER_EXPOSICION_PATH = os.path.join(MODEL_DIR, 'scaler_exposicion.pkl')
MODELO_DESENFOQUE_PATH = os.path.join(MODEL_DIR, 'modelo_desenfoque.pkl')
SCALER_DESENFOQUE_PATH = os.path.join(MODEL_DIR, 'scaler_desenfoque.pkl')
FEATURES_DESENFOQUE_PATH = os.path.join(MODEL_DIR, 'features_desenfoque.pkl')
FEATURES_EXPOSICION_PATH = os.path.join(MODEL_DIR, 'features_exposicion.pkl')

N_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))

THRESHOLD_HIGH_ENTROPY = 0.7  # 80% de entropía
THRESHOLD_MED_ENTROPY = 0.6  # 60% de entropía
THRESHOLD_HIGH_BLUR = 0.60  # 60% de probabilidad de borroso
THRESHOLD_MED_BLUR = 0.30   # 40% de probabilidad de borroso
DIFF_PROB = 0.2            # Diferencia significativa

OPTIMIZAR_HIPERPARAMETROS = True
ENTRENAR_NUEVO_MODELO = True