import os, hashlib, cv2, pandas as pd
from tqdm import tqdm
import xgboost as xgb
import numpy as np
import joblib
from config import *
from models.desenfoque import *
from models.exposicion import *

def entrenar_y_guardar_modelo_exp(df):
    
    print("🎯 Entrenando modelo de exposición...")
    modelo_exposicion = ModeloExposicion()
    modelo_exposicion.entrenar(df)
    
    return modelo_exposicion.model

def entrenar_y_guardar_modelo_des(df):
    print("\n🎯 Entrenando modelo de desenfoque...")
    modelo_desenfoque = ModeloDesenfoque()
    modelo_desenfoque.entrenar(df)
    
    return modelo_desenfoque.model

def actualizar_dataset_entrenamiento(dataset_dir, guardar_csv=True):
    
    CSV_DESENFOQUE_PATH = os.path.join(dataset_dir, 'dataset_desenfoque.csv')
    CSV_EXPOSICION_PATH = os.path.join(dataset_dir, 'dataset_exposicion.csv')
    # Cargar datasets existentes o crear nuevos (sin cambios)
    if os.path.exists(CSV_DESENFOQUE_PATH):
        df_desenfoque_existente = pd.read_csv(CSV_DESENFOQUE_PATH)
        print(f"✅ Dataset desenfoque existente cargado con {len(df_desenfoque_existente)} imágenes")
    else:
        df_desenfoque_existente = pd.DataFrame()
        print("🆕 Creando nuevo dataset de desenfoque")
    
    if os.path.exists(CSV_EXPOSICION_PATH):
        df_exposicion_existente = pd.read_csv(CSV_EXPOSICION_PATH)
        print(f"✅ Dataset exposición existente cargado con {len(df_exposicion_existente)} imágenes")
    else:
        df_exposicion_existente = pd.DataFrame()
        print("🆕 Creando nuevo dataset de exposición")
    
    # Identificar imágenes ya procesadas (sin cambios)
    imagenes_procesadas = set()
    if not df_desenfoque_existente.empty and 'file_hash' in df_desenfoque_existente.columns:
        imagenes_procesadas.update(df_desenfoque_existente['file_hash'])
    if not df_exposicion_existente.empty and 'file_hash' in df_exposicion_existente.columns:
        imagenes_procesadas.update(df_exposicion_existente['file_hash'])
    
    print(f"  - {len(imagenes_procesadas)} imágenes ya procesadas en total")
    
    # Recolectar nuevas imágenes (sin cambios)
    datos_desenfoque = []
    datos_exposicion = []
    nuevas_imagenes = 0
    
    # Mapeo de carpetas a tipos de clase - VALORES ACTUALIZADOS
    CARPETAS_CLASES = {
        'borrosa': ('desenfoque', 0),      # Actualizado
        'aceptable': ('desenfoque', 1),    # Actualizado
        'nitida': ('desenfoque', 2),       # Actualizado
        'subexpuesta': ('exposicion', 0),  # Actualizado
        'sobreexpuesta': ('exposicion', 0),
        'correcta': ('exposicion', 1) # Actualizado
    }
    
    for carpeta_nombre in CARPETAS_CLASES.keys():
        carpeta_dir = os.path.join(dataset_dir, carpeta_nombre)
        if not os.path.exists(carpeta_dir):
            print(f"⚠️ Advertencia: No se encontró directorio {carpeta_dir}")
            continue
        
        archivos = [f for f in os.listdir(carpeta_dir) 
                   if f.lower().endswith(('.jpg'))]
        
        print(f"📂 Verificando {len(archivos)} imágenes en {carpeta_nombre}...")
        
        for archivo in tqdm(archivos, desc=carpeta_nombre):
            img_path = os.path.join(carpeta_dir, archivo)
            
            # Calcular hash para verificar duplicados (sin cambios)
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            except Exception:
                print(f"❌ Error leyendo archivo: {img_path}")
                continue
                
            # Saltar si ya procesado (sin cambios)
            if file_hash in imagenes_procesadas:
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Error leyendo imagen: {img_path}")
                continue

            # >>>> ASIGNAR tipo_clase y valor_clase AQUÍ <<<<
            tipo_clase, valor_clase = CARPETAS_CLASES[carpeta_nombre]

            try:
                # Calcular métricas (sin cambios)
                if tipo_clase in ['desenfoque']:
                    metricas = calcular_metricas_desenfoque(img)
                elif tipo_clase in ['exposicion']:
                    metricas = calcular_metricas_exposicion(img)
                
                # Añadir metadatos (sin cambios)
                height, width = img.shape[:2]
                file_size = os.path.getsize(img_path) / 1024  # KB
                
                datos_base = {
                    'archivo': archivo,
                    'ruta': img_path,
                    'carpeta_origen': carpeta_nombre,
                    'ancho': width,
                    'alto': height,
                    'tamano_kb': file_size,
                    'file_hash': file_hash,
                    **metricas
                }
                
                # Determinar a qué dataset(s) pertenece (sin cambios en lógica, solo valores)
                tipo_clase, valor_clase = CARPETAS_CLASES[carpeta_nombre]
                
                if tipo_clase in ['desenfoque', 'ambos']:
                    datos_desenfoque.append({
                        **datos_base,
                        'clase': valor_clase,
                        'clase_nombre': list(CLASES_DESENFOQUE.keys())[list(CLASES_DESENFOQUE.values()).index(valor_clase)]
                    })
                
                if tipo_clase in ['exposicion', 'ambos']:
                    datos_exposicion.append({
                        **datos_base,
                        'clase': valor_clase,
                        'clase_nombre': list(CLASES_EXPOSICION.keys())[list(CLASES_EXPOSICION.values()).index(valor_clase)]
                    })
                
                nuevas_imagenes += 1
                
            except Exception as e:
                print(f"❌ Error procesando {img_path}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Combinar con datasets existentes (sin cambios)
    if datos_desenfoque:
        df_desenfoque_nuevo = pd.DataFrame(datos_desenfoque)
        df_desenfoque_actualizado = pd.concat([df_desenfoque_existente, df_desenfoque_nuevo], ignore_index=True)
        
        if guardar_csv:
            df_desenfoque_actualizado.to_csv(CSV_DESENFOQUE_PATH, index=False)
            print(f"💾 Dataset desenfoque actualizado: {len(df_desenfoque_actualizado)} imágenes")
    else:
        df_desenfoque_actualizado = df_desenfoque_existente
    
    if datos_exposicion:
        df_exposicion_nuevo = pd.DataFrame(datos_exposicion)
        df_exposicion_actualizado = pd.concat([df_exposicion_existente, df_exposicion_nuevo], ignore_index=True)
        
        if guardar_csv:
            df_exposicion_actualizado.to_csv(CSV_EXPOSICION_PATH, index=False)
            print(f"💾 Dataset exposición actualizado: {len(df_exposicion_actualizado)} imágenes")
    else:
        df_exposicion_actualizado = df_exposicion_existente
    
    print(f"🆕 Total nuevas imágenes procesadas: {nuevas_imagenes}")
    
    return df_desenfoque_actualizado, df_exposicion_actualizado

def calcular_score_final(pred_desenfoque, pred_exposicion):
    """
    Calcula score final 0-100 basado en ambas predicciones
    """
    # Convertir nombres de clases a valores numéricos
    valor_des = CLASES_DESENFOQUE[pred_desenfoque]
    valor_exp = CLASES_EXPOSICION[pred_exposicion]
    
    # Calcular score ponderado
    score_bruto = (valor_des * PESO_DESENFOQUE + 
                   valor_exp * PESO_EXPOSICION)
    
    # Normalizar a 0-100
    max_score = (max(CLASES_DESENFOQUE.values()) * PESO_DESENFOQUE + 
                 max(CLASES_EXPOSICION.values()) * PESO_EXPOSICION)
    
    return round((score_bruto / max_score) * 100, 1)

def interpretar_score(score):
    """Provee interpretación cualitativa del score"""
    if score >= 90: return "Excelente"
    elif score >= 75: return "Muy bueno"
    elif score >= 60: return "Aceptable"
    elif score >= 40: return "Necesita mejora"
    else: return "Deficiente"

def agregar_scores_dataset(df):
    """
    Agrega columnas de score a un dataset existente
    Útil para análisis de datos de entrenamiento
    """
    df = df.copy()
    
    # Calcular scores para cada fila
    scores = []
    interpretaciones = []
    
    for _, row in df.iterrows():
        try:
            score = calcular_score_final(row['clase_nombre_desenfoque'], 
                                       row['clase_nombre_exposicion'])
            interp = interpretar_score(score)
        except:
            score = 0
            interp = "Error"
        
        scores.append(score)
        interpretaciones.append(interp)
    
    df['score_final'] = scores
    df['interpretacion'] = interpretaciones
    
    return df

def calcular_metricas_seleccionadas(imagen_bgr, tipo='completo'):
    """
    Función wrapper que calcula métricas según el tipo requerido
    tipo: 'completo', 'desenfoque', 'exposicion'
    """
    if tipo == 'desenfoque':
        return calcular_metricas_desenfoque(imagen_bgr)
    elif tipo == 'exposicion':
        return calcular_metricas_exposicion(imagen_bgr)
    else:
        # Combinar ambas para compatibilidad
        metricas_des = calcular_metricas_desenfoque(imagen_bgr)
        metricas_exp = calcular_metricas_exposicion(imagen_bgr)
        return {**metricas_des, **metricas_exp}

def cargar_recursos_modelos():
    """
    Carga ambos modelos (desenfoque y exposición) con sus respectivos recursos
    """
    # Paths para modelo de desenfoque
    MODELO_DES_PATH = os.path.join(MODEL_DIR, 'modelo_desenfoque.pkl')
    SCALER_DES_PATH = os.path.join(MODEL_DIR, 'scaler_desenfoque.pkl')
    FEATURES_DES_PATH = os.path.join(MODEL_DIR, 'features_desenfoque.pkl')
    
    # Paths para modelo de exposición
    MODELO_EXP_PATH = os.path.join(MODEL_DIR, 'modelo_exposicion.pkl')
    SCALER_EXP_PATH = os.path.join(MODEL_DIR, 'scaler_exposicion.pkl')
    FEATURES_EXP_PATH = os.path.join(MODEL_DIR, 'features_exposicion.pkl')
    
    # Verificar que existen todos los archivos
    recursos_desenfoque = [MODELO_DES_PATH, SCALER_DES_PATH, FEATURES_DES_PATH]
    recursos_exposicion = [MODELO_EXP_PATH, SCALER_EXP_PATH, FEATURES_EXP_PATH]
    
    if not all(os.path.exists(p) for p in recursos_desenfoque):
        raise FileNotFoundError("❌ Faltan archivos del modelo de DESENFOQUE. Ejecuta el entrenamiento primero")
    
    if not all(os.path.exists(p) for p in recursos_exposicion):
        raise FileNotFoundError("❌ Faltan archivos del modelo de EXPOSICIÓN. Ejecuta el entrenamiento primero")
    
    # Cargar modelo de desenfoque
    modelo_des = xgb.Booster()
    modelo_des.load_model(MODELO_DES_PATH)
    scaler_des = joblib.load(SCALER_DES_PATH)
    feats_des = joblib.load(FEATURES_DES_PATH)
    
    # Cargar modelo de exposición
    modelo_exp = xgb.Booster()
    modelo_exp.load_model(MODELO_EXP_PATH)
    scaler_exp = joblib.load(SCALER_EXP_PATH)
    feats_exp = joblib.load(FEATURES_EXP_PATH)
    
    print(f"✅ Modelo desenfoque cargado: {MODELO_DES_PATH}")
    print(f"✅ Modelo exposición cargado: {MODELO_EXP_PATH}")
    
    return {
        'desenfoque': {
            'modelo': modelo_des,
            'scaler': scaler_des,
            'features': feats_des
        },
        'exposicion': {
            'modelo': modelo_exp,
            'scaler': scaler_exp,
            'features': feats_exp
        }
    }
