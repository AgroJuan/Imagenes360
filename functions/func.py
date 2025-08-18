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
from .model import calcular_metricas_seleccionadas
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    
    
    print(f"🗂 SHP con simbología guardado en: {shp_out}")

    return gdf

