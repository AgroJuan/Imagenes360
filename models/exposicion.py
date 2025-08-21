# models/exposure.py
from .base_model import BaseModel
from config import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
import cv2
import warnings

class ModeloExposicion(BaseModel):
    def __init__(self):
        super().__init__(MODELO_EXPOSICION_PATH, SCALER_EXPOSICION_PATH, FEATURES_EXPOSICION_PATH)
        self.features_seleccionados = [
            'mean_brightness', 'colorfulness', 'rms_contrast', 'michelson_contrast',
            'brightness_std', 'saturation_mean', 'value_mean',
            'white_clipping', 'black_clipping', 'exposure_balance', 'histogram_skew'
        ]

    def entrenar(self, df):
        """Entrenamiento específico para exposición (silencioso en warnings y con nombres de features)."""
        X_scaled, y = self.preprocess_data(df, self.features_seleccionados)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Hiperparámetros
        if OPTIMIZAR_HIPERPARAMETROS:
            best_params, best_score = self.optimizar_hiperparametros(X_train, y_train)
            print(f"⚙️ Mejores parámetros exposición: {best_params}")
            print(f"⚙️ Mejor precisión exposición: {best_score:.4f}")
        else:
            best_params = {'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}

        # XGBoost con DMatrix y nombres de columnas para evitar warnings
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.features_seleccionados)
        dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=self.features_seleccionados)

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

        # Evaluación
        val_pred_probs = self.model.predict(dval)
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"🔍 Precisión exposición en validación: {val_acc:.4f}")

        # Guardar
        self.guardar_modelo()
        print(f"💾 Modelo exposición guardado en {self.model_path}")

        return self.model


def calcular_metricas_exposicion(imagen_bgr):
    """Calcula métricas para exposición, evitando divisiones por cero y warnings numéricos."""
    # Conversión de color
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    metricas = {}
    eps = 1e-6

    # 1) Brillo
    metricas['mean_brightness'] = float(np.mean(gray))

    # 2) Desv. estándar del brillo
    metricas['brightness_std'] = float(np.std(gray))

    # 3) Contraste RMS
    g_mean = float(gray.mean())
    metricas['rms_contrast'] = float(np.sqrt(np.mean((gray.astype(np.float32) - g_mean) ** 2)))

    # 4) Contraste de Michelson
    max_val = float(gray.max())
    min_val = float(gray.min())
    metricas['michelson_contrast'] = float((max_val - min_val) / (max_val + min_val + eps))

    # 5) Colorfulness (Hasler & Süsstrunk)
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    rg = R - G
    yb = 0.5 * (R + G) - B
    metricas['colorfulness'] = float(
        np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    )

    # 6) HSV (saturación y valor)
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    metricas['saturation_mean'] = float(np.mean(hsv[:, :, 1].astype(np.float32)))
    metricas['value_mean']      = float(np.mean(hsv[:, :, 2].astype(np.float32)))

    # 7) Clipping
    gray_f = gray.astype(np.float32)
    white_clipped = float(np.mean(gray_f > 245.0))
    black_clipped = float(np.mean(gray_f < 10.0))
    metricas['white_clipping'] = white_clipped
    metricas['black_clipping'] = black_clipped

    # 8) Balance de exposición
    bright_pixels = gray_f[gray_f > 128.0]
    dark_pixels   = gray_f[gray_f <= 128.0]
    if bright_pixels.size > 0 and dark_pixels.size > 0:
        metricas['exposure_balance'] = float(np.mean(bright_pixels) - np.mean(dark_pixels))
    else:
        metricas['exposure_balance'] = 0.0

    # 9) Asimetría del histograma (skewness robusta con eps en la std)
    std = float(gray_f.std())
    denom = (std ** 3) + eps
    metricas['histogram_skew'] = float(np.mean((gray_f - gray_f.mean()) ** 3) / denom) if denom != 0 else 0.0

    return metricas
