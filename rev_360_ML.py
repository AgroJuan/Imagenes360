
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
