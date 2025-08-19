# models/modelo_desenfoque.py
from .base_model import BaseModel
from config import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

class ModeloDesenfoque(BaseModel):
    def __init__(self):
        super().__init__(MODELO_DESENFOQUE_PATH, SCALER_DESENFOQUE_PATH, FEATURES_DESENFOQUE_PATH)
        self.features_seleccionados = [
            'glcm_homogeneity', 'entropy', 'freq_ratio', 'ssim', 'glcm_contrast',
            'blur_extent', 'laplacian_variance', 'fft_energy'
        ]
    
    def entrenar(self, df):
        """Entrenamiento específico para desenfoque"""
        X_scaled, y = self.preprocess_data(df, self.features_seleccionados)
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Optimizar hiperparámetros
        if OPTIMIZAR_HIPERPARAMETROS:
            best_params, best_score = self.optimizar_hiperparametros(X_train, y_train)
            print(f"⚙️ Mejores parámetros desenfoque: {best_params}")
            print(f"⚙️ Mejor precisión desenfoque: {best_score:.4f}")
        else:
            best_params = {'max_depth': 4, 'learning_rate': 0.2, 'subsample': 0.6}
        
        # Entrenar modelo
        self.model = self.entrenar_modelo(X_train, y_train, X_val, y_val, best_params)
        
        # Evaluar
        val_pred_probs = self.model.predict(xgb.DMatrix(X_val))
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"🔍 Precisión desenfoque en validación: {val_acc:.4f}")
        
        # Guardar
        self.guardar_modelo()
        print(f"💾 Modelo desenfoque guardado en {self.model_path}")
        
        return self.model
    
def calcular_metricas_desenfoque(imagen_bgr):
    """Calcula métricas especializadas para detección de desenfoque"""
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    metricas = {}
    
    # 1. GLCM (textura - muy importante para desenfoque)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    metricas['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    metricas['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # 2. Entropía (indicador de información/textura)
    metricas['entropy'] = shannon_entropy(gray)
    
    # 3. Análisis de frecuencia FFT
    fft = np.fft.fft2(gray.astype(float))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    cy, cx = gray.shape[0]//2, gray.shape[1]//2
    
    # Ratio alta/baja frecuencia
    high_freq = magnitude[cy-15:cy+15, cx-15:cx+15]
    low_freq = magnitude.copy()
    low_freq[cy-25:cy+25, cx-25:cx+25] = 0
    mean_low = np.mean(low_freq[low_freq > 0]) or 1e-6
    metricas['freq_ratio'] = np.mean(high_freq) / mean_low
    
    # 4. Energía FFT (nueva métrica)
    metricas['fft_energy'] = np.sum(magnitude**2) / (gray.size)  # Energía normalizada
    
    # 5. SSIM (similitud estructural)
    small = cv2.resize(gray, (256, 256))
    blurred = cv2.GaussianBlur(small, (5, 5), 0.5)
    metricas['ssim'] = ssim(small, blurred)
    
    # 6. Detección de bordes y blur
    edges = cv2.Canny(gray, 50, 150)
    if np.any(edges):
        edge_gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
        metricas['blur_extent'] = np.mean(edge_gradient[edges > 0])
    else:
        metricas['blur_extent'] = 0.0
    
    # 7. Varianza Laplaciana (nueva métrica - excelente para blur detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metricas['laplacian_variance'] = np.var(laplacian)
    
    # 8. ELIMINAR métricas no requeridas:
    # 'rms_contrast' fue eliminada según tu lista
    
    return metricas