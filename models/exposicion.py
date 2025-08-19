# models/exposure.py
from .base_model import BaseModel
from config import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
import cv2

class ModeloExposicion(BaseModel):
    def __init__(self):
        super().__init__(MODELO_EXPOSICION_PATH, SCALER_EXPOSICION_PATH, FEATURES_EXPOSICION_PATH)
        self.features_seleccionados = [
            'mean_brightness', 'colorfulness', 'rms_contrast', 'michelson_contrast',
            'brightness_std', 'saturation_mean', 'value_mean',
            'white_clipping', 'black_clipping', 'exposure_balance', 'histogram_skew'
        ]
    
    def entrenar(self, df):
        """Entrenamiento específico para exposición"""
        X_scaled, y = self.preprocess_data(df, self.features_seleccionados)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Optimizar hiperparámetros
        if OPTIMIZAR_HIPERPARAMETROS:
            best_params, best_score = self.optimizar_hiperparametros(X_train, y_train)
            print(f"⚙️ Mejores parámetros exposición: {best_params}")
            print(f"⚙️ Mejor precisión exposición: {best_score:.4f}")
        else:
            best_params = {'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
        
        # Entrenar modelo
        self.model = self.entrenar_modelo(X_train, y_train, X_val, y_val, best_params)
        
        # Evaluar
        val_pred_probs = self.model.predict(xgb.DMatrix(X_val))
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"🔍 Precisión exposición en validación: {val_acc:.4f}")
        
        # Guardar
        self.guardar_modelo()
        print(f"💾 Modelo exposición guardado en {self.model_path}")
        
        return self.model
    
def calcular_metricas_exposicion(imagen_bgr):
    """Calcula métricas especializadas para detección de problemas de exposición"""
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    metricas = {}
    
    # 1. Brillo y luminosidad (fundamental para exposición)
    metricas['mean_brightness'] = np.mean(gray)
    
    # 2. Desviación estándar del brillo (nueva métrica)
    metricas['brightness_std'] = np.std(gray)
    
    # 3. Contraste RMS (nueva métrica - reemplaza histogram_skew)
    metricas['rms_contrast'] = np.sqrt(np.mean((gray - gray.mean())**2))
    
    # 4. Contraste de Michelson (ya existe)
    max_val = float(gray.max())
    min_val = float(gray.min())
    metricas['michelson_contrast'] = (max_val - min_val) / (max_val + min_val + 1e-6)
    
    # 5. Colorfulness (los colores se saturan con mala exposición)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    rg = R.astype(float) - G.astype(float)
    yb = 0.5*(R.astype(float) + G.astype(float)) - B.astype(float)
    metricas['colorfulness'] = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    # 6. Saturación y valor en HSV
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    metricas['saturation_mean'] = np.mean(hsv[:, :, 1])
    metricas['value_mean'] = np.mean(hsv[:, :, 2])
    
    # 7. Detección de clipping (píxeles quemados/sobreexpuestos)
    white_clipped = np.mean(gray > 245)  # Porcentaje de píxeles casi blancos
    black_clipped = np.mean(gray < 10)   # Porcentaje de píxeles casi negros
    metricas['white_clipping'] = white_clipped
    metricas['black_clipping'] = black_clipped
    
    # 8. Balance de exposición (nueva métrica)
    # Dividir la imagen en áreas claras y oscuras
    bright_pixels = gray[gray > 128]
    dark_pixels = gray[gray <= 128]
    
    if len(bright_pixels) > 0 and len(dark_pixels) > 0:
        metricas['exposure_balance'] = np.mean(bright_pixels) - np.mean(dark_pixels)
    else:
        metricas['exposure_balance'] = 0.0
    
    metricas['histogram_skew'] = np.mean((gray - gray.mean())**3) / (gray.std()**3 + 1e-6)
    
    return metricas