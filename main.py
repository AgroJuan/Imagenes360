from config.config import *
from functions.func import *
from models.model import *
from functions.func import *
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but RobustScaler was fitted with feature names")

if __name__ == '__main__':
    # Verificar dependencia crítica
    try:
        import py360convert
    except ImportError:
        print("❌ ERROR: py360convert no instalado. Ejecuta: pip install py360convert")
        exit(1)
    
    # 1) Actualizar dataset de entrenamiento
    try:
        df_desenfoque, df_exposicion = actualizar_dataset_entrenamiento(TRAIN_DIR)
    except Exception as e:
        print(f"❌ Error actualizando dataset: {str(e)}")
        exit(1)
    
    # Verificar datasets
    if df_desenfoque.empty and df_exposicion.empty:
        print("❌ Error: Datasets de entrenamiento vacíos")
        exit()
    
    # 2) Entrenar modelos si es necesario
    debe_entrenar = not all(os.path.exists(p) for p in [
        MODELO_EXPOSICION_PATH, SCALER_EXPOSICION_PATH, FEATURES_EXPOSICION_PATH,
        MODELO_DESENFOQUE_PATH, SCALER_DESENFOQUE_PATH, FEATURES_DESENFOQUE_PATH
    ]) or ENTRENAR_NUEVO_MODELO
    
    if debe_entrenar:
        print("🛠️  Entrenando modelos...")
        try:
            entrenar_y_guardar_modelo_des(df_desenfoque)
            entrenar_y_guardar_modelo_exp(df_exposicion)
            print("✅ Modelos entrenados y guardados exitosamente")
            # Entrena ambos modelos
        except Exception as e:
            print(f"❌ Error en entrenamiento: {str(e)}")
            exit(1)
    else:
        print("✅ Modelos existentes. No se necesita reentrenamiento")
    
    # 3) Cargar recursos de ambos modelos
    try:
        recursos = cargar_recursos_modelos()
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
        
        # 4.1 Predecir imágenes con ambos modelos
        try:
            procesar_imagenes_test(TEST_DIR, recursos)
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
