from .config.config import *
from .functions.func import *
from .functions.model import *

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
