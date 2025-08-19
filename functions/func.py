import os, hashlib, cv2, pandas as pd, gc
import py360convert
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import multiprocessing
import xgboost as xgb
from models.desenfoque import *
from models.exposicion import *
from models.model import *
from functions.func import *
from config import *
from concurrent.futures import ThreadPoolExecutor, as_completed


def procesar_imagenes_test(test_dir, recursos):
    """
    Procesa todas las imágenes nuevas en un directorio de test usando ambos modelos
    y genera salida combinada con scoring final.
    
    recursos: diccionario con ambos modelos (de cargar_recursos_modelos())
    """
    print(f"\n📂 Procesando: {test_dir}")
    CSV_PRED = os.path.join(test_dir, 'resultados_prediccion.csv')
    
    # Preparar columnas del CSV
    columnas = ['archivo', 'score_final', 'calidad']
    for cara in FACE_NAMES:
        columnas += [
            f'cara_{cara}_desenfoque', f'cara_{cara}_exposicion',
            f'prob_{cara}_borrosa', f'prob_{cara}_aceptable', f'prob_{cara}_nitida',
            f'prob_{cara}_subexpuesta', f'prob_{cara}_correcta', f'prob_{cara}_sobreexpuesta',
            f'score_{cara}'
        ]
    
    # Crear archivo si no existe
    if not os.path.exists(CSV_PRED):
        pd.DataFrame(columns=columnas).to_csv(CSV_PRED, index=False)
    
    # Procesar imágenes
    archivos = [f for f in os.listdir(test_dir) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Filtrar imágenes ya procesadas
    if os.path.exists(CSV_PRED):
        try:
            df_existente = pd.read_csv(CSV_PRED)
            archivos_procesados = set(df_existente['archivo'])
            archivos = [f for f in archivos if f not in archivos_procesados]
            print(f"  - {len(archivos)} imágenes nuevas por procesar")
        except Exception as e:
            print(f"⚠️ Error leyendo CSV existente: {e}. Procesando todas las imágenes")
    
    if not archivos:
        print("✅ Todas las imágenes ya procesadas. Saltando...")
        return
    
    lock = multiprocessing.Lock()
    
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = []
        for archivo in archivos:
            futures.append(
                executor.submit(
                    predecir_cubemap_combinado, 
                    archivo, recursos, CSV_PRED, lock, test_dir
                )
            )
        
        # Barra de progreso
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicción combinada"):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error procesando imagen: {e}")
    
    print(f"💾 Resultados guardados en {CSV_PRED}")
def predecir_cubemap_combinado(archivo, recursos, csv_path, lock, test_dir):
    """
    Predice usando ambos modelos y guarda resultados combinados
    """
    try:
        img_path = os.path.join(test_dir, archivo)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ No se pudo leer: {archivo}")
            return
        
        # Extraer caras del cubemap (asumiendo que tienes esta función)
        caras = extraer_caras_cubemap(img)
        
        resultados_caras = []
        
        for i, cara_img in enumerate(caras):
            cara_nombre = FACE_NAMES[i]
            
            # Extraer features para ambos modelos
            features_des = calcular_metricas_desenfoque(cara_img)
            features_exp = calcular_metricas_exposicion(cara_img)
            
            # Preprocesar features
            features_des_array = np.array([features_des[feat] for feat in recursos['desenfoque']['features']])
            features_exp_array = np.array([features_exp[feat] for feat in recursos['exposicion']['features']])
            
            # Escalar
            features_des_scaled = recursos['desenfoque']['scaler'].transform([features_des_array])
            features_exp_scaled = recursos['exposicion']['scaler'].transform([features_exp_array])
            
            # Predecir
            pred_des = recursos['desenfoque']['modelo'].predict(xgb.DMatrix(features_des_scaled))[0]
            pred_exp = recursos['exposicion']['modelo'].predict(xgb.DMatrix(features_exp_scaled))[0]
            
            # Obtener clases predichas
            clase_des = np.argmax(pred_des)
            clase_exp = np.argmax(pred_exp)
            
            # Convertir a nombres de clases
            nombres_clases_des = list(CLASES_DESENFOQUE.keys())
            nombres_clases_exp = list(CLASES_EXPOSICION.keys())
            
            pred_nombre_des = nombres_clases_des[clase_des]
            pred_nombre_exp = nombres_clases_exp[clase_exp]
            
            # Calcular score para esta cara
            score_cara = calcular_score_final(pred_nombre_des, pred_nombre_exp)
            
            resultados_caras.append({
                'cara': cara_nombre,
                'pred_desenfoque': pred_nombre_des,
                'pred_exposicion': pred_nombre_exp,
                'probs_desenfoque': pred_des.tolist(),
                'probs_exposicion': pred_exp.tolist(),
                'score': score_cara
            })
        
        # Calcular score global (promedio de todas las caras)
        score_global = np.mean([r['score'] for r in resultados_caras])
        calidad_global = interpretar_score(score_global)
        
        # Preparar datos para CSV
        fila_data = {
            'archivo': archivo,
            'score_final': score_global,
            'calidad': calidad_global
        }
        
        # Añadir datos por cara
        for resultado in resultados_caras:
            cara = resultado['cara']
            
            fila_data[f'cara_{cara}_desenfoque'] = resultado['pred_desenfoque']
            fila_data[f'cara_{cara}_exposicion'] = resultado['pred_exposicion']
            fila_data[f'score_{cara}'] = resultado['score']
            
            # Probabilidades desenfoque
            probs_des = resultado['probs_desenfoque']
            for j, clase in enumerate(['borrosa', 'aceptable', 'nitida']):
                fila_data[f'prob_{cara}_{clase}'] = probs_des[j]
            
            # Probabilidades exposición
            probs_exp = resultado['probs_exposicion']
            for j, clase in enumerate(['subexpuesta', 'correcta', 'sobreexpuesta']):
                fila_data[f'prob_{cara}_{clase}'] = probs_exp[j]
        
        # Guardar en CSV (thread-safe)
        with lock:
            df_existente = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=fila_data.keys())
            df_nuevo = pd.DataFrame([fila_data])
            df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)
            df_actualizado.to_csv(csv_path, index=False)
            
    except Exception as e:
        print(f"❌ Error procesando {archivo}: {str(e)}")
        import traceback
        traceback.print_exc()
def analizar_resultados_csv(csv_pred_path, csv_rev_path,
                          th_high_entropy=THRESHOLD_HIGH_ENTROPY,
                          th_med_entropy=THRESHOLD_MED_ENTROPY,
                          th_high_blur=THRESHOLD_HIGH_BLUR,
                          th_med_blur=THRESHOLD_MED_BLUR,
                          diff_prob=DIFF_PROB):
    """
    Analiza el CSV de predicción con ambos modelos y guarda resultados con clasificación global.
    Ahora considera tanto desenfoque como exposición en el análisis.
    """
    df = pd.read_csv(csv_pred_path)
    
    resultados = []
    for _, fila in df.iterrows():
        nombre = fila['archivo']
        
        # Inicializar listas para ambas clasificaciones
        clases_des, clases_exp = [], []
        probs_b, probs_a, probs_n = [], [], []  # Desenfoque
        probs_sub, probs_corr, probs_sob = [], [], []  # Exposición
        scores_caras = []

        for f in FACE_NAMES:
            # Clasificaciones por cara
            clases_des.append(fila[f'cara_{f}_desenfoque'])
            clases_exp.append(fila[f'cara_{f}_exposicion'])
            
            # Probabilidades desenfoque
            probs_b.append(fila[f'prob_{f}_borrosa'])
            probs_a.append(fila[f'prob_{f}_aceptable'])
            probs_n.append(fila[f'prob_{f}_nitida'])
            
            # Probabilidades exposición
            probs_sub.append(fila[f'prob_{f}_subexpuesta'])
            probs_corr.append(fila[f'prob_{f}_correcta'])
            probs_sob.append(fila[f'prob_{f}_sobreexpuesta'])
            
            # Scores por cara
            scores_caras.append(fila[f'score_{f}'])

        # Promedios de probabilidad por clase (desenfoque)
        mean_b = np.mean(probs_b)
        mean_a = np.mean(probs_a)
        mean_n = np.mean(probs_n)
        
        # Promedios de probabilidad por clase (exposición)
        mean_sub = np.mean(probs_sub)
        mean_corr = np.mean(probs_corr)
        mean_sob = np.mean(probs_sob)
        
        # Score global
        score_global = fila['score_final'] if 'score_final' in fila else np.mean(scores_caras)
        calidad_global = fila['calidad'] if 'calidad' in fila else interpretar_score(score_global)

        # Entropía de la distribución por cara (desenfoque)
        dist_des = np.array([probs_b, probs_a, probs_n])  # 3 x num_caras
        entropias_des = [entropy(dist_des[:, i]) for i in range(dist_des.shape[1])]
        mean_entropy_des = np.mean(entropias_des)
        
        # Entropía de la distribución por cara (exposición)
        dist_exp = np.array([probs_sub, probs_corr, probs_sob])  # 3 x num_caras
        entropias_exp = [entropy(dist_exp[:, i]) for i in range(dist_exp.shape[1])]
        mean_entropy_exp = np.mean(entropias_exp)

        # Clasificación global (mayor promedio para cada modelo)
        idx_des = np.argmax([mean_b, mean_a, mean_n])
        clase_global_des = ['borrosa', 'aceptable', 'nitida'][idx_des]
        
        idx_exp = np.argmax([mean_sub, mean_corr, mean_sob])
        clase_global_exp = ['subexpuesta', 'correcta', 'sobreexpuesta'][idx_exp]

        # Consistencia entre caras (cuántas caras coinciden con la clasificación global)
        consistencia_des = sum(1 for c in clases_des if c == clase_global_des) / len(clases_des)
        consistencia_exp = sum(1 for c in clases_exp if c == clase_global_exp) / len(clases_exp)

        # Marcar si necesita revisión - CRITERIOS ACTUALIZADOS
        # Criterios para desenfoque
        cond1_des = (mean_entropy_des > th_high_entropy) and (mean_b > th_med_blur)
        cond2_des = (mean_b > th_med_blur) and (
                    (abs(mean_b - mean_a) <= diff_prob) or 
                    (abs(mean_b - mean_n) <= diff_prob)
                )
        cond3_des = (mean_b > th_high_blur)
        cond4_des = (clase_global_des == 'borrosa' and mean_entropy_des > th_med_entropy) or \
                   (clase_global_des == 'aceptable' and mean_entropy_des > th_high_entropy)
        
        # Criterios para exposición
        cond1_exp = (mean_entropy_exp > th_high_entropy) and (mean_sub > 0.3 or mean_sob > 0.3)
        cond2_exp = (abs(mean_sub - mean_corr) <= diff_prob) or (abs(mean_sob - mean_corr) <= diff_prob)
        cond3_exp = consistencia_exp < 0.5  # Baja consistencia entre caras
        
        # Criterios combinados
        cond5 = score_global < 60  # Score bajo global
        cond6 = consistencia_des < 0.5  # Baja consistencia en desenfoque
        
        necesita_revision = (cond1_des or cond2_des or cond3_des or cond4_des or 
                           cond1_exp or cond2_exp or cond3_exp or cond5 or cond6)

        resultados.append({
            'archivo': nombre,
            'score_global': round(score_global, 1),
            'calidad_global': calidad_global,
            
            # Clasificaciones globales
            'clase_global_desenfoque': clase_global_des,
            'clase_global_exposicion': clase_global_exp,
            
            # Probabilidades promedio desenfoque
            'mean_prob_borrosa': round(mean_b, 3),
            'mean_prob_aceptable': round(mean_a, 3),
            'mean_prob_nitida': round(mean_n, 3),
            
            # Probabilidades promedio exposición
            'mean_prob_subexpuesta': round(mean_sub, 3),
            'mean_prob_correcta': round(mean_corr, 3),
            'mean_prob_sobreexpuesta': round(mean_sob, 3),
            
            # Entropías
            'mean_entropy_desenfoque': round(mean_entropy_des, 3),
            'mean_entropy_exposicion': round(mean_entropy_exp, 3),
            
            # Consistencias
            'consistencia_desenfoque': round(consistencia_des, 2),
            'consistencia_exposicion': round(consistencia_exp, 2),
            
            # Flags de revisión
            'revisar': necesita_revision,
            'revisar_desenfoque': cond1_des or cond2_des or cond3_des or cond4_des,
            'revisar_exposicion': cond1_exp or cond2_exp or cond3_exp,
            'revisar_score': cond5,
            'revisar_consistencia': cond6
        })

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(csv_rev_path, index=False)
    
    # Estadísticas del análisis
    total = len(df_out)
    por_revisar = len(df_out[df_out['revisar']])
    
    print(f"📊 Análisis completado:")
    print(f"   - Total imágenes: {total}")
    print(f"   - Necesitan revisión: {por_revisar} ({por_revisar/total*100:.1f}%)")
    print(f"   - Guardado en: {csv_rev_path}")
    
    return df_out, df_out[df_out['revisar']]

def extraer_caras_cubemap(imagen_cubemap):

    height, width = imagen_cubemap.shape[:2]
    
    # Verificar que la imagen tiene el ratio esperado para 4 caras horizontales
    # Ratio esperado: ancho = 4 * alto (para 4 caras cuadradas)
    expected_ratio = 4/1  # 4 caras de ancho, 1 cara de alto
    actual_ratio = width / height
    
    if not (3.8 <= actual_ratio <= 4.2):
        print(f"⚠️  Ratio inesperado: {actual_ratio:.2f} (esperado ~4.0). Ajustando automáticamente.")
    
    # Calcular tamaño de cada cara (asumiendo que son cuadradas)
    cara_size = height  # Como hay 1 fila de caras, el alto define el tamaño
    
    # Asegurarse de que el ancho es múltiplo de 4
    if width % 4 != 0:
        # Redimensionar para que sea divisible por 4
        new_width = (width // 4) * 4
        imagen_cubemap = cv2.resize(imagen_cubemap, (new_width, height))
        width = new_width
        cara_size = width // 4
        print(f"⚠️  Imagen redimensionada a {width}x{height} para extracción precisa")
    
    # Extraer las 4 caras horizontales
    caras = []
    
    # Orden de extracción: Izquierda, Frontal, Derecha, Trasera
    for i in range(4):
        x_start = i * cara_size
        x_end = (i + 1) * cara_size
        
        cara = imagen_cubemap[0:height, x_start:x_end]
        caras.append(cara)
    
    # Reordenar para el orden estándar: Frontal, Derecha, Trasera, Izquierda
    # El extraction order es: [0:Izquierda, 1:Frontal, 2:Derecha, 3:Trasera]
    # Queremos: [Frontal, Derecha, Trasera, Izquierda]
    orden_correcto = [1, 2, 3, 0]  # Frontal, Derecha, Trasera, Izquierda
    
    caras_ordenadas = [caras[i] for i in orden_correcto]
    
    return caras_ordenadas
# %% FUNCIONES DE GRAFICADO
def plot_dataset_quality(df_all, output_dir):
    """
    Genera gráficos para entender la 'salud' del dataset con ambos modelos:
      - Barplot de recuento de imágenes por clase de desenfoque
      - Barplot de recuento de imágenes por clase de exposición  
      - Scatter de score final vs entropía
      - Distribución de scores finales
    """
    # Paletas de colores para ambos modelos
    palette_desenfoque = {
        'borrosa': '#d62728',
        'aceptable': '#ff7f0e', 
        'nitida': '#2ca02c'
    }
    
    palette_exposicion = {
        'subexpuesta': '#8c564b',
        'correcta': '#17becf',
        'sobreexpuesta': '#e377c2'
    }
    
    # 1. Gráfico de distribución de clases de DESENFOQUE
    plt.figure(figsize=(8, 5))
    if 'clase_nombre_desenfoque' in df_all.columns:
        conteo_des = df_all['clase_nombre_desenfoque'].value_counts().reset_index()
        conteo_des.columns = ['clase', 'recuento']
        total_des = len(df_all)
        
        # Ordenar según clases predefinidas
        clases_orden_des = ['borrosa', 'aceptable', 'nitida']
        conteo_des = conteo_des.set_index('clase').reindex(clases_orden_des).reset_index()
        
        ax1 = sns.barplot(
            x='clase', y='recuento', data=conteo_des,
            hue='clase', palette=palette_desenfoque, dodge=False, legend=False
        )
        
        # Añadir etiquetas con valores y porcentajes
        for i, row in enumerate(conteo_des.itertuples()):
            porcentaje = 100 * row.recuento / total_des
            texto = f"{row.recuento}\n({porcentaje:.1f}%)"
            ax1.text(i, row.recuento + 0.5, texto, ha='center', va='bottom', fontsize=9)
        
        plt.title(f"Distribución por calidad de DESENFOQUE (n={total_des})")
        plt.xlabel("Clasificación de desenfoque")
        plt.ylabel("Número de imágenes")
        plt.ylim(0, max(conteo_des['recuento']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'barplot_desenfoque.png'), dpi=600)
        plt.close()
    
    # 2. Gráfico de distribución de clases de EXPOSICIÓN
    plt.figure(figsize=(8, 5))
    if 'clase_nombre_exposicion' in df_all.columns:
        conteo_exp = df_all['clase_nombre_exposicion'].value_counts().reset_index()
        conteo_exp.columns = ['clase', 'recuento']
        total_exp = len(df_all)
        
        # Ordenar según clases predefinidas
        clases_orden_exp = ['subexpuesta', 'correcta', 'sobreexpuesta']
        conteo_exp = conteo_exp.set_index('clase').reindex(clases_orden_exp).reset_index()
        
        ax2 = sns.barplot(
            x='clase', y='recuento', data=conteo_exp,
            hue='clase', palette=palette_exposicion, dodge=False, legend=False
        )
        
        # Añadir etiquetas con valores y porcentajes
        for i, row in enumerate(conteo_exp.itertuples()):
            porcentaje = 100 * row.recuento / total_exp
            texto = f"{row.recuento}\n({porcentaje:.1f}%)"
            ax2.text(i, row.recuento + 0.5, texto, ha='center', va='bottom', fontsize=9)
        
        plt.title(f"Distribución por calidad de EXPOSICIÓN (n={total_exp})")
        plt.xlabel("Clasificación de exposición")
        plt.ylabel("Número de imágenes")
        plt.ylim(0, max(conteo_exp['recuento']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'barplot_exposicion.png'), dpi=600)
        plt.close()
    
    # 3. Distribución de SCORES FINALES (si existe la columna)
    plt.figure(figsize=(10, 6))
    if 'score_final' in df_all.columns:
        # Histograma + KDE
        sns.histplot(df_all['score_final'], bins=20, kde=True, alpha=0.7)
        
        # Añadir líneas verticales para umbrales de calidad
        plt.axvline(90, color='green', linestyle='--', alpha=0.7, label='Excelente (90+)')
        plt.axvline(75, color='blue', linestyle='--', alpha=0.7, label='Muy bueno (75+)')
        plt.axvline(60, color='orange', linestyle='--', alpha=0.7, label='Aceptable (60+)')
        plt.axvline(40, color='red', linestyle='--', alpha=0.7, label='Deficiente (<40)')
        
        plt.title(f"Distribución de Scores Finales (n={len(df_all)})")
        plt.xlabel("Score de calidad (0-100)")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'histogram_scores.png'), dpi=600)
        plt.close()
    
    # 4. Scatter plot: Score final vs Entropía (si existen ambas columnas)
    plt.figure(figsize=(8, 6))
    if 'score_final' in df_all.columns and 'entropy' in df_all.columns:
        ax4 = sns.scatterplot(
            x='score_final', y='entropy',
            data=df_all,
            alpha=0.6,
            s=60
        )
        
        # Añadir líneas de referencia
        plt.axvline(60, color='red', linestyle='--', alpha=0.5, label='Umbral aceptable (60)')
        if 'THRESHOLD_HIGH_ENTROPY' in globals():
            plt.axhline(THRESHOLD_HIGH_ENTROPY, color='gray', linestyle='--', label='Alta entropía')
        
        plt.title("Score Final vs Entropía")
        plt.xlabel("Score de calidad (0-100)")
        plt.ylabel("Entropía")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_score_vs_entropy.png'), dpi=600)
        plt.close()
    
    # 5. Matriz de correlación entre métricas clave (opcional)
    plt.figure(figsize=(10, 8))
    metricas_interes = [
        'score_final', 'entropy', 'glcm_contrast', 'glcm_homogeneity',
        'mean_brightness', 'colorfulness', 'rms_contrast'
    ]
    
    metricas_disponibles = [m for m in metricas_interes if m in df_all.columns]
    
    if len(metricas_disponibles) >= 3:
        corr_matrix = df_all[metricas_disponibles].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title("Matriz de Correlación entre Métricas")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=600)
        plt.close()
    
    # 6. Boxplot de scores por clase de desenfoque
    plt.figure(figsize=(10, 6))
    if 'score_final' in df_all.columns and 'clase_nombre_desenfoque' in df_all.columns:
        sns.boxplot(
            x='clase_nombre_desenfoque', 
            y='score_final', 
            data=df_all,
            palette=palette_desenfoque,
            order=['borrosa', 'aceptable', 'nitida']
        )
        plt.title("Distribución de Scores por Calidad de Desenfoque")
        plt.xlabel("Clasificación de desenfoque")
        plt.ylabel("Score final")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplot_scores_desenfoque.png'), dpi=600)
        plt.close()
    
    print(f"🎨 Gráficos de calidad del dataset guardados en {output_dir}")
# %% ACTUALIZAR Y FILTRAR SHP
def actualizar_y_filtrar_shp(shp_path, csv_revision, test_dir, shp_out):
    """
    Actualiza shapefile con información de calidad y categorías del nuevo sistema.
    Ahora usa el score final y la categoría de calidad en lugar de clases simples.
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

    # 4) Leer CSV de revisión completo
    if not os.path.exists(csv_revision):
        raise FileNotFoundError(f"❌ No se encontró el CSV de revisión: {csv_revision}")
    df_rev = pd.read_csv(csv_revision)

    # Verificar columnas necesarias para el nuevo sistema
    required_cols = ['archivo', 'score_global', 'calidad_global', 
                    'clase_global_desenfoque', 'clase_global_exposicion',
                    'revisar', 'revisar_desenfoque', 'revisar_exposicion']
    
    # Columnas mínimas requeridas
    min_required = ['archivo', 'score_global', 'calidad_global', 'revisar']
    
    if not set(min_required).issubset(df_rev.columns):
        missing = set(min_required) - set(df_rev.columns)
        raise ValueError(f"❌ Faltan columnas esenciales en CSV: {missing}")
    
    # 5) Unir información de clasificación al shapefile (left join)
    available_cols = [col for col in required_cols if col in df_rev.columns]
    gdf = gdf.merge(
        df_rev[available_cols], 
        on='archivo', 
        how='left',
        suffixes=('', '_csv')
    )
    
    # 6) Rellenar valores nulos para registros no procesados
    gdf['score_global'] = gdf['score_global'].fillna(-1)
    gdf['calidad_global'] = gdf['calidad_global'].fillna('No procesado')
    gdf['revisar'] = gdf['revisar'].fillna(False)
    
    # Rellenar otras columnas opcionales si existen
    for col in ['clase_global_desenfoque', 'clase_global_exposicion',
               'revisar_desenfoque', 'revisar_exposicion']:
        if col in gdf.columns:
            if col.startswith('clase'):
                gdf[col] = gdf[col].fillna('No procesado')
            else:
                gdf[col] = gdf[col].fillna(False)
    
    # 7) Añadir columnas de simbología para categorías
    gdf = agregar_simbologia_calidad(gdf)
    
    # 8) Ordenar por score (mejores primero) para mejor visualización
    gdf = gdf.sort_values('score_global', ascending=False)
    
    # 9) Exportar shapefile
    gdf.to_file(shp_out, driver='ESRI Shapefile')
    
    # 10) Generar archivo de estilos QGIS (.qml) automáticamente
    generar_archivo_estilos_qgis(shp_out, gdf)
    
    print(f"🗂 SHP con simbología guardado en: {shp_out}")
    print(f"🎨 Archivo de estilos QGIS generado: {shp_out.replace('.shp', '.qml')}")
    
    # Estadísticas
    stats = calcular_estadisticas_calidad(gdf)
    print(f"📊 Estadísticas de calidad:")
    for categoria, count in stats.items():
        print(f"   - {categoria}: {count} imágenes")
    
    return gdf

def agregar_simbologia_calidad(gdf):
    """
    Añade columnas de simbología para visualización en GIS
    """
    # Mapeo de categorías a colores (formato RGB hexadecimal)
    colores_calidad = {
        'Excelente': '#00FF00',      # Verde
        'Muy bueno': '#A6FF00',      # Verde claro
        'Aceptable': '#FFFF00',      # Amarillo
        'Necesita mejora': '#FF7F00', # Naranja
        'Deficiente': '#FF0000',     # Rojo
        'No procesado': '#808080'    # Gris
    }
    
    # Mapeo de categorías a tamaños de símbolo
    tamanos_calidad = {
        'Excelente': 6,
        'Muy bueno': 5,
        'Aceptable': 4,
        'Necesita mejora': 3,
        'Deficiente': 2,
        'No procesado': 1
    }
    
    # Añadir columnas de simbología
    gdf['color'] = gdf['calidad_global'].map(colores_calidad)
    gdf['tamano'] = gdf['calidad_global'].map(tamanos_calidad)
    
    # Colores para problemas específicos
    gdf['color_revision'] = '#FF00FF'  # Magenta para revisión
    gdf['color_desenfoque'] = '#FF0000'  # Rojo para problemas desenfoque
    gdf['color_exposicion'] = '#0000FF'  # Azul para problemas exposición
    
    return gdf

def generar_archivo_estilos_qgis(shp_path, gdf):
    """
    Genera archivo .qml automáticamente para estilos en QGIS
    """
    qml_content = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0-Białowieża" styleCategories="Symbology">
  <renderer-v2 symbollevels="0" type="singleSymbol" forceraster="0" enableorderby="0">
    <symbols>
      <symbol name="0" force_rhr="0" type="marker" clip_to_extent="1" alpha="1">
        <data_defined_properties>
          <Option type="Map">
            <Option name="name" type="QString" value=""/>
            <Option name="properties"/>
            <Option name="type" type="QString" value="collection"/>
          </Option>
        </data_defined_properties>
        <layer locked="0" pass="0" class="SimpleMarker" enabled="1">
          <Option type="Map">
            <Option name="angle" type="QString" value="0"/>
            <Option name="cap_style" type="QString" value="square"/>
            <Option name="color" type="QString" value="[% "color" %]"/>
            <Option name="horizontal_anchor_point" type="QString" value="1"/>
            <Option name="joinstyle" type="QString" value="bevel"/>
            <Option name="name" type="QString" value="circle"/>
            <Option name="offset" type="QString" value="0,0"/>
            <Option name="offset_map_unit_scale" type="QString" value="3x:0,0,0,0,0,0"/>
            <Option name="offset_unit" type="QString" value="MM"/>
            <Option name="outline_color" type="QString" value="35,35,35,255"/>
            <Option name="outline_style" type="QString" value="solid"/>
            <Option name="outline_width" type="QString" value="0.2"/>
            <Option name="outline_width_map_unit_scale" type="QString" value="3x:0,0,0,0,0,0"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="scale_method" type="QString" value="diameter"/>
            <Option name="size" type="QString" value="[% "tamano" %]"/>
            <Option name="size_map_unit_scale" type="QString" value="3x:0,0,0,0,0,0"/>
            <Option name="size_unit" type="QString" value="MM"/>
            <Option name="vertical_anchor_point" type="QString" value="1"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value=""/>
              <Option name="properties" type="Map">
                <Option name="fillColor" type="Map">
                  <Option name="active" type="QString" value="1"/>
                  <Option name="expression" type="QString" value="color"/>
                  <Option name="type" type="QString" value="3"/>
                </Option>
                <Option name="size" type="Map">
                  <Option name="active" type="QString" value="1"/>
                  <Option name="expression" type="QString" value="tamano"/>
                  <Option name="type" type="QString" value="3"/>
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
    <rotation/>
    <sizescale/>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>0</layerGeometryType>
</qgis>'''
    
    qml_path = shp_path.replace('.shp', '.qml')
    with open(qml_path, 'w', encoding='utf-8') as f:
        f.write(qml_content)

def calcular_estadisticas_calidad(gdf):
    """
    Calcula estadísticas de distribución de calidad
    """
    stats = gdf['calidad_global'].value_counts().to_dict()
    
    # Asegurar que todas las categorías estén presentes
    categorias_completas = {
        'Excelente': 0,
        'Muy bueno': 0,
        'Aceptable': 0,
        'Necesita mejora': 0,
        'Deficiente': 0,
        'No procesado': 0
    }
    
    for cat in categorias_completas:
        if cat in stats:
            categorias_completas[cat] = stats[cat]
        else:
            categorias_completas[cat] = 0
    
    return categorias_completas