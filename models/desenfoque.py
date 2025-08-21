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
import warnings

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

        # Entrenar modelo (con nombres de columnas para evitar warnings)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.features_seleccionados)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=self.features_seleccionados)

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            **best_params
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dval, "eval")],
            early_stopping_rounds=20,
            verbose_eval=10
        )

        # Evaluar
        val_pred_probs = self.model.predict(dval)
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

    # 1. GLCM (textura)
    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        metricas['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        metricas['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
    except Exception:
        metricas['glcm_contrast'] = 0
        metricas['glcm_homogeneity'] = 0

    # 2. Entropía
    try:
        metricas['entropy'] = float(shannon_entropy(gray))
    except Exception:
        metricas['entropy'] = 0

    # 3. Análisis de frecuencia FFT
    fft = np.fft.fft2(gray.astype(float))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))  # log1p evita warnings de log(0)

    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    high_freq = magnitude[cy-15:cy+15, cx-15:cx+15]
    low_freq = magnitude.copy()
    low_freq[cy-25:cy+25, cx-25:cx+25] = 0

    mean_low = np.mean(low_freq[low_freq > 0]) if np.any(low_freq > 0) else 1e-6
    metricas['freq_ratio'] = float(np.mean(high_freq) / mean_low)

    # 4. Energía FFT
    metricas['fft_energy'] = float(np.sum(magnitude**2) / gray.size)

    # 5. SSIM (silenciar warnings si la imagen es plana)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            small = cv2.resize(gray, (256, 256))
            blurred = cv2.GaussianBlur(small, (5, 5), 0.5)
            metricas['ssim'] = float(ssim(small, blurred, data_range=small.max() - small.min() + 1e-6))
        except Exception:
            metricas['ssim'] = 0

    # 6. Blur extent
    try:
        edges = cv2.Canny(gray, 50, 150)
        if np.any(edges):
            edge_gradient = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
            metricas['blur_extent'] = float(np.mean(edge_gradient[edges > 0]))
        else:
            metricas['blur_extent'] = 0.0
    except Exception:
        metricas['blur_extent'] = 0

    # 7. Varianza Laplaciana
    try:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metricas['laplacian_variance'] = float(np.var(laplacian))
    except Exception:
        metricas['laplacian_variance'] = 0

    return metricas
