# %%
# %% LIBRERÍAS
import os
import cv2
import gc
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import py360convert
import geopandas as gpd
from tqdm import tqdm
from scipy.stats import entropy
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import warnings

# Ignorar warnings específicos
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# %%
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

# %%
# %% FUNCIÓN DE MÉTRICAS SELECCIONADAS
def calcular_metricas_seleccionadas(imagen_bgr):
    """Calcula solo las métricas seleccionadas optimizadas"""
    # Convertir a RGB y escala de grises
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    metricas = {}
    
    # 1. GLCM (Contraste y Homogeneidad)
    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        metricas['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        metricas['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    except Exception:
        metricas['glcm_contrast'] = np.nan
        metricas['glcm_homogeneity'] = np.nan
    
    # 2. Entropía
    metricas['entropy'] = shannon_entropy(gray)
    
    # 3. Frecuencia ratio (FFT optimizado)
    try:
        fft = np.fft.fft2(gray.astype(float))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        cy, cx = gray.shape[0]//2, gray.shape[1]//2
        high_freq = magnitude[cy-15:cy+15, cx-15:cx+15]
        low_freq = magnitude.copy()
        low_freq[cy-25:cy+25, cx-25:cx+25] = 0
        mean_low = np.mean(low_freq[low_freq > 0]) or 1e-6
        metricas['freq_ratio'] = np.mean(high_freq) / mean_low
    except Exception:
        metricas['freq_ratio'] = np.nan
    
    # 4. SSIM (con versión redimensionada para eficiencia)
    try:
        small = cv2.resize(gray, (256, 256))
        blurred = cv2.GaussianBlur(small, (5, 5), 0.5)
        metricas['ssim'] = ssim(small, blurred)
    except Exception:
        metricas['ssim'] = np.nan
    
    # 5. Brillo y Contraste
    metricas['mean_brightness'] = np.mean(gray)
    max_val = float(gray.max())
    min_val = float(gray.min())
    metricas['michelson_contrast'] = (max_val - min_val) / (max_val + min_val + 1e-6)
    metricas['rms_contrast'] = np.sqrt(np.mean((gray - gray.mean())**2))
    
    # 6. Colorfulness
    try:
        R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        rg = R.astype(float) - G.astype(float)
        yb = 0.5*(R.astype(float) + G.astype(float)) - B.astype(float)
        metricas['colorfulness'] = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    except Exception:
        metricas['colorfulness'] = np.nan
    
    # 7. Blur Extent (optimizado)
    try:
        edges = cv2.Canny(gray, 50, 150)
        if np.any(edges):
            edge_gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
            metricas['blur_extent'] = np.mean(edge_gradient[edges > 0])
        else:
            metricas['blur_extent'] = 0.0
    except Exception:
        metricas['blur_extent'] = np.nan
        
    return metricas

# %% ENTRENAMIENTO CORREGIDO
def entrenar_y_guardar_modelo(df):
    """Entrena con validación y escalado usando API correcta de XGBoost"""
    FEATS_SELECCIONADOS = [
        'glcm_homogeneity', 'entropy', 'freq_ratio', 'ssim', 'glcm_contrast',
        'mean_brightness', 'colorfulness', 'blur_extent', 'rms_contrast', 'michelson_contrast'
    ]
    
    # Preprocesamiento
    X = df[FEATS_SELECCIONADOS]
    y = df['clase']
    
    # Escalado robusto
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optimización de hiperparámetros
    best_params = {}
    if OPTIMIZAR_HIPERPARAMETROS:
        print("⚙️ Optimizando hiperparámetros...")
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        grid = GridSearchCV(
            xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_scaled, y)
        best_params = grid.best_params_
        print(f"⚙️ Mejores parámetros: {best_params}")
        print(f"⚙️ Mejor precisión: {grid.best_score_:.4f}")
    else:
        best_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
    
    # Dividir datos para validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convertir a formato DMatrix de XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Parámetros del modelo
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'seed': 42,
        **best_params
    }
    
    # Entrenar con early stopping usando API de entrenamiento
    modelo = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'eval')],
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    # Evaluar en validación
    val_pred_probs = modelo.predict(dval)
    val_pred = np.argmax(val_pred_probs, axis=1)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"🔍 Precisión en validación: {val_acc:.4f}")
    
    # Guardar modelo y escalador
    modelo.save_model(MODELO_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATS_SELECCIONADOS, FEATURES_PATH)
    print(f"💾 Modelo y escalador guardados en {MODELO_PATH}")
    
    return modelo

# %% PREDICCIÓN POR CUBEMAP (OPTIMIZADA)
def predecir_cubemap(archivo, modelo, scaler, feats, csv_out, lock, test_dir):
    img_path = os.path.join(test_dir, archivo)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error leyendo imagen: {img_path}")
        return
    
    try:
        # Convertir a cubemap (1024x1024 por cara)
        cubemap = py360convert.e2c(img, face_w=1024, cube_format='list')
        
        # VERIFICAR SI ES UN CÓDIGO DE ERROR (ENTERO)
        if isinstance(cubemap, int):
            raise ValueError(f"Error en conversión cubemap (código {cubemap})")
        
        # Tomar solo las 4 caras laterales (F, R, B, L)
        caras_seleccionadas = cubemap[:4]
        fila = {'archivo': archivo}
        
        for i, face in enumerate(caras_seleccionadas):
            nombre_cara = FACE_NAMES[i]
            metricas = calcular_metricas_seleccionadas(face)
            
            # Preparar datos para predicción
            X_pred = pd.DataFrame([metricas.get(f, 0) for f in feats], index=feats).T
            
            # Escalar características
            X_pred_scaled = scaler.transform(X_pred)
            
            # Convertir a DMatrix
            d_pred = xgb.DMatrix(X_pred_scaled)
            
            # Predecir probabilidades
            probs = modelo.predict(d_pred)[0]
            pred = CLASES_INV[np.argmax(probs)]
            
            # Almacenar resultados
            fila[f'cara_{nombre_cara}'] = pred
            fila[f'prob_{nombre_cara}_borrosa'] = round(probs[0], 3)
            fila[f'prob_{nombre_cara}_aceptable'] = round(probs[1], 3)
            fila[f'prob_{nombre_cara}_nitida'] = round(probs[2], 3)
        
        # Guardar resultados
        with lock:
            pd.DataFrame([fila]).to_csv(csv_out, mode='a', index=False, header=False)
            
    except Exception as e:
        print(f"❌ Error procesando {archivo}: {str(e)}")
    finally:
        # Liberar memoria
        del img
        if 'cubemap' in locals():
            del cubemap
        gc.collect()

# %% FUNCIÓN PARA CONSTRUIR DATASET DE ENTRENAMIENTO
def construir_dataset_entrenamiento(dataset_dir, guardar_csv=True):
    """
    Construye el dataset de entrenamiento desde las subcarpetas: borrosa, aceptable, nitida
    """
    # Configuración
    CLASES_ENTRENAMIENTO = {
        'borrosa': 0,
        'aceptable': 1,
        'nitida': 2
    }
    CSV_PATH = os.path.join(dataset_dir, 'dataset_entrenamiento.csv')
    
    # Si ya existe el CSV, cargarlo directamente
    if os.path.exists(CSV_PATH):
        print(f"✅ Dataset de entrenamiento ya existe. Cargando desde {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    
    print("🛠️ Construyendo dataset de entrenamiento desde subcarpetas...")
    datos = []
    total_imagenes = 0
    processed_hashes = set()
    
    for clase_nombre, clase_val in CLASES_ENTRENAMIENTO.items():
        clase_dir = os.path.join(dataset_dir, clase_nombre)
        if not os.path.exists(clase_dir):
            print(f"⚠️ Advertencia: No se encontró directorio {clase_dir}")
            continue
        
        archivos = [f for f in os.listdir(clase_dir) 
                   if f.lower().endswith(('.jpg'))]
        
        print(f"📂 Procesando {len(archivos)} imágenes en {clase_nombre}...")
        
        for archivo in tqdm(archivos, desc=clase_nombre):
            img_path = os.path.join(clase_dir, archivo)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"❌ Error leyendo imagen: {img_path}")
                continue
            
            # Calcular hash para evitar duplicados
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in processed_hashes:
                print(f"⏩ Saltando duplicado: {archivo}")
                continue
                
            processed_hashes.add(file_hash)
            
            try:
                # Calcular métricas
                metricas = calcular_metricas_seleccionadas(img)
                
                # Añadir metadatos
                height, width = img.shape[:2]
                file_size = os.path.getsize(img_path) / 1024  # KB
                
                datos.append({
                    'archivo': archivo,
                    'ruta': img_path,
                    'clase': clase_val,
                    'clase_nombre': clase_nombre,
                    'ancho': width,
                    'alto': height,
                    'tamano_kb': file_size,
                    'file_hash': file_hash,
                    **metricas
                })
                total_imagenes += 1
                
            except Exception as e:
                print(f"❌ Error procesando {img_path}: {str(e)}")
    
    # Crear DataFrame
    df = pd.DataFrame(datos)
    
    # Guardar si se solicita
    if guardar_csv and not df.empty:
        df.to_csv(CSV_PATH, index=False)
        print(f"💾 Dataset guardado en {CSV_PATH} ({total_imagenes} imágenes)")
    
    return df

# %% FUNCIÓN PARA ACTUALIZAR DATASET DE ENTRENAMIENTO
def actualizar_dataset_entrenamiento(dataset_dir, guardar_csv=True):
    """
    Actualiza el dataset de entrenamiento con nuevas imágenes desde las subcarpetas,
    omitiendo las ya procesadas.
    """
    # Configuración
    CLASES_ENTRENAMIENTO = {
        'borrosa': 0,
        'aceptable': 1,
        'nitida': 2
    }
    CSV_PATH = os.path.join(dataset_dir, 'dataset_entrenamiento.csv')
    
    # Cargar dataset existente o crear uno nuevo
    if os.path.exists(CSV_PATH):
        df_existente = pd.read_csv(CSV_PATH)
        print(f"✅ Dataset existente cargado con {len(df_existente)} imágenes")
    else:
        df_existente = pd.DataFrame()
        print("🆕 Creando nuevo dataset de entrenamiento")
    
    # Identificar imágenes ya procesadas
    imagenes_procesadas = set()
    if not df_existente.empty and 'file_hash' in df_existente.columns:
        imagenes_procesadas = set(df_existente['file_hash'])
        print(f"  - {len(imagenes_procesadas)} imágenes ya procesadas")
    
    # Recolectar nuevas imágenes
    datos_nuevos = []
    nuevas_imagenes = 0
    
    for clase_nombre, clase_val in CLASES_ENTRENAMIENTO.items():
        clase_dir = os.path.join(dataset_dir, clase_nombre)
        if not os.path.exists(clase_dir):
            print(f"⚠️ Advertencia: No se encontró directorio {clase_dir}")
            continue
        
        archivos = [f for f in os.listdir(clase_dir) 
                   if f.lower().endswith(('.jpg'))]
        
        print(f"📂 Verificando {len(archivos)} imágenes en {clase_nombre}...")
        
        for archivo in tqdm(archivos, desc=clase_nombre):
            img_path = os.path.join(clase_dir, archivo)
            
            # Calcular hash para verificar duplicados
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            except Exception:
                print(f"❌ Error leyendo archivo: {img_path}")
                continue
                
            # Saltar si ya procesado
            if file_hash in imagenes_procesadas:
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Error leyendo imagen: {img_path}")
                continue
            
            try:
                # Calcular métricas
                metricas = calcular_metricas_seleccionadas(img)
                
                # Añadir metadatos
                height, width = img.shape[:2]
                file_size = os.path.getsize(img_path) / 1024  # KB
                
                datos_nuevos.append({
                    'archivo': archivo,
                    'ruta': img_path,
                    'clase': clase_val,
                    'clase_nombre': clase_nombre,
                    'ancho': width,
                    'alto': height,
                    'tamano_kb': file_size,
                    'file_hash': file_hash,
                    **metricas
                })
                nuevas_imagenes += 1
                
            except Exception as e:
                print(f"❌ Error procesando {img_path}: {str(e)}")
    
    # Combinar con dataset existente
    if datos_nuevos:
        df_nuevo = pd.DataFrame(datos_nuevos)
        df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
        
        # Guardar si se solicita
        if guardar_csv:
            df_actualizado.to_csv(CSV_PATH, index=False)
            print(f"🆕 Añadidas {nuevas_imagenes} nuevas imágenes")
            print(f"💾 Dataset actualizado con {len(df_actualizado)} imágenes en total")
    else:
        df_actualizado = df_existente
        print("✅ No se encontraron nuevas imágenes para añadir")
    
    return df_actualizado

# %% ANÁLISIS DE RESULTADOS
def analizar_resultados_csv(csv_pred_path, csv_rev_path,
                            th_high_entropy=THRESHOLD_HIGH_ENTROPY,
                            th_med_entropy=THRESHOLD_MED_ENTROPY,
                            th_high_blur=THRESHOLD_HIGH_BLUR,
                            th_med_blur=THRESHOLD_MED_BLUR,
                            diff_prob=DIFF_PROB):
    """Analiza el CSV de predicción y guarda resultados con clasificación global."""
    df = pd.read_csv(csv_pred_path)

    resultados = []
    for _, fila in df.iterrows():
        nombre = fila['archivo']
        probs_b, probs_a, probs_n = [], [], []
        clases = []

        for f in FACE_NAMES:
            clases.append(fila[f'cara_{f}'])
            probs_b.append(fila[f'prob_{f}_borrosa'])
            probs_a.append(fila[f'prob_{f}_aceptable'])
            probs_n.append(fila[f'prob_{f}_nitida'])

        # Promedios de probabilidad por clase
        mean_b = np.mean(probs_b)
        mean_a = np.mean(probs_a)
        mean_n = np.mean(probs_n)

        # Entropía de la distribución por cara
        dist = np.array([probs_b, probs_a, probs_n])  # 3 x 4
        entropias = [entropy(dist[:, i]) for i in range(dist.shape[1])]
        mean_entropy = np.mean(entropias)

        # Clasificación global (mayor promedio)
        idx = np.argmax([mean_b, mean_a, mean_n])
        clase_global = ['borrosa', 'aceptable', 'nitida'][idx]

        # Marcar si necesita revisión
        cond1 = (mean_entropy > th_high_entropy) and (mean_b > th_med_blur)
        cond2 = (mean_b > th_med_blur) and (
                    (abs(mean_b - mean_a) <= diff_prob) or 
                    (abs(mean_b - mean_n) <= diff_prob)
                )
        cond3 = (mean_b > th_high_blur)
        cond4 = (clase_global == 'borrosa' and mean_entropy > th_med_entropy) or (clase_global == 'aceptable' and mean_entropy > th_high_entropy)
        
        necesita_revision = cond1 or cond2 or cond3 or cond4

        resultados.append({
            'archivo': nombre,
            'clase': clase_global,
            'mean_prob_borrosa': round(mean_b, 3),
            'mean_prob_aceptable': round(mean_a, 3),
            'mean_prob_nitida': round(mean_n, 3),
            'mean_entropy': round(mean_entropy, 3),
            'revisar': necesita_revision
        })

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_rev_path, index=False)
    print(f"📊 Análisis completado y guardado en {csv_rev_path}")
    return df_out, df_out[df_out['revisar']]

# %% FUNCIONES DE GRAFICADO
def plot_dataset_quality(df_all, output_dir):
    """
    Genera dos gráficos para entender la 'salud' del dataset:
      - Barplot de recuento de imagenes por clase global.
      - Scatter de mean_prob_borrosa vs. mean_entropy, coloreado por clase_global.
    """
    # Violin plot
    plt.figure(figsize=(6,4))
    
    palette = {
        'borrosa':   '#d62728',
        'aceptable': '#ff7f0e',
        'nitida':    '#2ca02c'
    }
    
    # Calcular recuentos por clase
    conteo = df_all['clase'].value_counts().reset_index()
    conteo.columns = ['clase', 'recuento']
    total = len(df_all)

    # Ordenar según clases predefinidas
    clases_orden = ['borrosa', 'aceptable', 'nitida']
    conteo = conteo.set_index('clase').reindex(clases_orden).reset_index()

    # Crear el barplot
    plt.figure(figsize=(8,5))
    ax = sns.barplot(
        x='clase', 
        y='recuento',
        data=conteo,
        hue='clase',
        palette=palette,
        dodge=False,  # Importante para evitar duplicación de barras
        legend=False
    )

    # Añadir etiquetas con valores y porcentajes
    for i, row in enumerate(conteo.itertuples()):
        porcentaje = 100 * row.recuento / total
        texto = f"{row.recuento}\n({porcentaje:.1f}%)"
        ax.text(i, row.recuento + 0.5, texto, 
                ha='center', va='bottom', fontsize=9)

    plt.title(f"Distribución de imágenes por calidad (n={len(df_all)})")
    plt.xlabel("Clasificación de calidad")
    plt.ylabel("Número de imágenes")
    plt.ylim(0, max(conteo['recuento']) * 1.15)  # Espacio para etiquetas
    plt.tight_layout()
    fp1 = os.path.join(output_dir, 'barplot_clasificacion.png')
    plt.savefig(fp1, dpi=600)
    plt.close()
    
    # Scatter plot entropía vs prob borrosa
    plt.figure(figsize=(7,5))
    ax = sns.scatterplot(
        x='mean_prob_borrosa', y='mean_entropy',
        hue='clase', style='revisar',
        data=df_all,
        palette=palette,
        legend=False,
        alpha=0.7,
        s=80
    )
    
    plt.axvline(THRESHOLD_HIGH_BLUR, color='gray', linestyle='--', label='Umbral borroso')
    plt.axhline(THRESHOLD_HIGH_ENTROPY, color='black', linestyle='--', label='Umbral entropía')
    plt.title("Entropía vs. Probabilidad borrosa media")
    plt.xlabel("Probabilidad borrosa media")
    plt.ylabel("Entropía media")
    fp2 = os.path.join(output_dir, 'scatter_entropy_vs_prob.png')
    plt.savefig(fp2, dpi=600)
    plt.close()

    print(f"🎨 Gráficos guardados en {output_dir}")

# %% ACTUALIZAR Y FILTRAR SHP
def actualizar_y_filtrar_shp(shp_path, csv_revision, test_dir, shp_out):
    """
    1) Lee un shapefile que tiene columna 'Path' con rutas completas a las imágenes.
    2) Reemplaza esas rutas por test_dir/<archivo> manteniendo solo el nombre.
    3) Lee csv_revision (que debe tener 'archivo', 'clase' y 'revisar'),
       mantiene todos los registros pero añade información de clasificación.
    4) Cruza con el GeoDataFrame original manteniendo todos los puntos.
    5) Crea un archivo de estilo QML (.qml) con la simbología predeterminada:
        - Gris: No requiere revisión
        - Verde: Revisión - Nitida
        - Naranja: Revisión - Aceptable
        - Rojo: Revisión - Borrosa
    6) Escribe el shapefile completo con las nuevas columnas y el archivo de estilo.
    """
    # 1) Leer shapefile
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo SHP: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # 2) Comprobar columna Path
    if 'Path' not in gdf.columns:
        raise ValueError("❌ El shapefile no contiene la columna 'Path'")

    # 3) Reemplazar rutas y extraer nombre de archivo
    gdf['archivo'] = gdf['Path'].apply(lambda p: os.path.basename(p))
    gdf['Path'] = gdf['archivo'].apply(lambda fn: os.path.join(test_dir, fn))

    # 4) Leer CSV de revisión completo (todos los registros)
    if not os.path.exists(csv_revision):
        raise FileNotFoundError(f"❌ No se encontró el CSV de revisión: {csv_revision}")
    df_rev = pd.read_csv(csv_revision)

    # Verificar columnas necesarias
    required_cols = ['archivo', 'clase', 'revisar']
    if not set(required_cols).issubset(df_rev.columns):
        missing = set(required_cols) - set(df_rev.columns)
        raise ValueError(f"❌ Faltan columnas en CSV: {missing}")
    
    # 5) Unir información de clasificación al shapefile (left join)
    gdf = gdf.merge(
        df_rev[required_cols], 
        on='archivo', 
        how='left',
        suffixes=('', '_csv')
    )
    
    # Rellenar valores nulos
    gdf['revisar'] = gdf['revisar'].fillna(False)
    gdf['clase'] = gdf['clase'].fillna('No procesado')
    
    # 6) Exportar shapefile
    gdf.to_file(shp_out, driver='ESRI Shapefile')
    
    # 7) Crear archivo QML con la simbología predeterminada
    qml_path = shp_out.replace('.shp', '.qml')
    crear_archivo_qml(qml_path)
    
    print(f"🗂 SHP con simbología guardado en: {shp_out}")
    print(f"🎨 Archivo de estilo QGIS generado: {qml_path}")
    
    return gdf

def crear_archivo_qml(qml_path):
    """
    Crea un QML muy básico (simbología categorizada) codificado en UTF-8,
    sin caracteres exóticos en el encabezado.
    """
    qml_content = """<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.22.0" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="clase">
    <categories>
      <category value="borrosa" label="Borrosa" symbol="0" render="true"/>
      <category value="aceptable" label="Aceptable" symbol="1" render="true"/>
      <category value="nitida" label="Nítida" symbol="2" render="true"/>
      <category value="No procesado" label="No procesado" symbol="3" render="true"/>
    </categories>
    <symbols>
      <!-- Borrosa: rojo -->
      <symbol type="marker" name="0">
        <layer class="SimpleMarker">
          <Option name="color" type="QString" value="215,25,28,255"/>
          <Option name="size"  type="QString" value="4"/>
        </layer>
      </symbol>
      <!-- Aceptable: naranja -->
      <symbol type="marker" name="1">
        <layer class="SimpleMarker">
          <Option name="color" type="QString" value="253,141,60,255"/>
          <Option name="size"  type="QString" value="4"/>
        </layer>
      </symbol>
      <!-- Nítida: verde -->
      <symbol type="marker" name="2">
        <layer class="SimpleMarker">
          <Option name="color" type="QString" value="51,160,44,255"/>
          <Option name="size"  type="QString" value="4"/>
        </layer>
      </symbol>
      <!-- No procesado: gris -->
      <symbol type="marker" name="3">
        <layer class="SimpleMarker">
          <Option name="color" type="QString" value="169,169,169,255"/>
          <Option name="size"  type="QString" value="3"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""
    # Escribimos siempre en UTF-8
    with open(qml_path, 'w', encoding='utf-8') as f:
        f.write(qml_content)


# %% CARGAR RECURSOS DEL MODELO
def cargar_recursos_modelo():
    """Carga modelo, scaler y features con verificación"""
    if not all(os.path.exists(p) for p in [MODELO_PATH, SCALER_PATH, FEATURES_PATH]):
        raise FileNotFoundError("❌ Faltan archivos del modelo. Ejecuta el entrenamiento primero")
    
    modelo = xgb.Booster()
    modelo.load_model(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    feats = joblib.load(FEATURES_PATH)
    
    print(f"✅ Modelo cargado: {MODELO_PATH}")
    return modelo, scaler, feats

# %% PROCESAR IMÁGENES DE TEST
def procesar_imagenes_test(test_dir, modelo, scaler, feats):
    """Procesa todas las imágenes nuevas en un directorio de test"""
    print(f"\n📂 Procesando: {test_dir}")
    CSV_PRED = os.path.join(test_dir, 'resultados_prediccion.csv')
    
    # Preparar archivo CSV
    columnas = ['archivo']
    for cara in FACE_NAMES:
        columnas += [f'cara_{cara}', f'prob_{cara}_borrosa', 
                   f'prob_{cara}_aceptable', f'prob_{cara}_nitida']
    
    # Crear archivo si no existe
    if not os.path.exists(CSV_PRED):
        pd.DataFrame(columns=columnas).to_csv(CSV_PRED, index=False)
    
    # Procesar imágenes
    archivos = [f for f in os.listdir(test_dir) 
               if f.lower().endswith(('.jpg'))]
    
    # Filtrar imágenes ya procesadas
    if os.path.exists(CSV_PRED):
        try:
            df_existente = pd.read_csv(CSV_PRED)
            archivos_procesados = set(df_existente['archivo'])
            archivos = [f for f in archivos if f not in archivos_procesados]
            print(f"  - {len(archivos)} imágenes nuevas por procesar")
        except:
            print("⚠️ Error leyendo CSV existente. Procesando todas las imágenes")
    
    if not archivos:
        print("✅ Todas las imágenes ya procesadas. Saltando...")
        return
    
    lock = multiprocessing.Lock()
    
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = []
        for archivo in archivos:
            futures.append(
                executor.submit(
                    predecir_cubemap, 
                    archivo, modelo, scaler, feats, CSV_PRED, lock, test_dir
                )
            )
        
        # Barra de progreso
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicción"):
            future.result()  # Capturar excepciones si las hay
    
    print(f"💾 Resultados guardados en {CSV_PRED}")

# %%
# %% MAIN
if __name__ == '__main__':
    # Verificar dependencia crítica
    try:
        import py360convert
    except ImportError:
        print("❌ ERROR: py360convert no instalado. Ejecuta: pip install py360convert")
        exit(1)
    
    # 1) Actualizar dataset de entrenamiento
    try:
        df_train = actualizar_dataset_entrenamiento(DATASET_DIR)
    except Exception as e:
        print(f"❌ Error actualizando dataset: {str(e)}")
        exit(1)
    
    if df_train.empty:
        print("❌ Error: Dataset de entrenamiento vacío")
        exit()
    
    # 2) Entrenar modelo si es necesario
    debe_entrenar = not all(os.path.exists(p) for p in [MODELO_PATH, SCALER_PATH, FEATURES_PATH]) or ENTRENAR_NUEVO_MODELO
    
    if debe_entrenar:
        print("🛠️  Entrenando modelo...")
        try:
            entrenar_y_guardar_modelo(df_train)
        except Exception as e:
            print(f"❌ Error en entrenamiento: {str(e)}")
            exit(1)
    else:
        print("✅ Modelo existente. No se necesita reentrenamiento")
    
    # 3) Cargar recursos del modelo
    try:
        modelo, scaler, feats = cargar_recursos_modelo()
    except Exception as e:
        print(str(e))
        exit(1)
    
    # 4) Procesar cada carpeta de test
    for TEST_DIR in TEST_DIRS:
        if not os.path.exists(TEST_DIR):
            print(f"⚠️ Directorio no encontrado: {TEST_DIR}")
            continue
            
        print(f"\n{'='*50}")
        print(f"📂 PROCESANDO DIRECTORIO: {TEST_DIR}")
        print(f"{'='*50}")
        
        # 4.1 Predecir imágenes
        try:
            procesar_imagenes_test(TEST_DIR, modelo, scaler, feats)
        except Exception as e:
            print(f"❌ Error procesando imágenes: {str(e)}")
            continue
            
        # 4.2 Analizar resultados
        CSV_PRED = os.path.join(TEST_DIR, 'resultados_prediccion.csv')
        CSV_REV = os.path.join(TEST_DIR, 'resultados_revisar.csv')
        
        if not os.path.exists(CSV_PRED):
            print(f"⚠️ No se encontraron resultados: {CSV_PRED}")
            continue
            
        try:
            df_all, df_rev = analizar_resultados_csv(CSV_PRED, CSV_REV)
            plot_dataset_quality(df_all, TEST_DIR)
        except Exception as e:
            print(f"❌ Error analizando resultados: {str(e)}")
            continue
            
        # 4.3 Procesar SHP si existe
        SHP_PATH = os.path.join(TEST_DIR, 'Centros_Fotos.shp')
        if os.path.exists(SHP_PATH):
            SHP_OUT = os.path.join(TEST_DIR, 'Centros_Fotos_Revision.shp')
            try:
                actualizar_y_filtrar_shp(SHP_PATH, CSV_REV, TEST_DIR, SHP_OUT)
            except Exception as e:
                print(f"❌ Error procesando SHP: {str(e)}")
    
    print("\n" + "="*50)
    print("✅✅✅ PROCESAMIENTO COMPLETADO CON ÉXITO ✅✅✅")
    print("="*50)
