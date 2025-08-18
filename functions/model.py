import os, hashlib, cv2, pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler


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
