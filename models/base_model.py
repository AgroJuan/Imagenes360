# models/base_model.py
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd

class BaseModel:
    def __init__(self, model_path, scaler_path, features_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.model = None
        self.scaler = RobustScaler()
        self.features_seleccionados = []
    
    def preprocess_data(self, df, features):
        """Preprocesamiento común"""
        X = df[features]
        y = df['clase']
        # Fit and transform using DataFrame to preserve feature names
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=features, index=X.index)
        return X_scaled, y
    
    def optimizar_hiperparametros(self, X, y):
        """Optimización común de hiperparámetros"""
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        grid = GridSearchCV(
            xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X, y)
        return grid.best_params_, grid.best_score_
    
    def entrenar_modelo(self, X_train, y_train, X_val, y_val, best_params):
        """Entrenamiento común"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'seed': 42,
            **best_params
        }
        
        modelo = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        return modelo
    
    def guardar_modelo(self):
        """Guardar modelo, scaler y features"""
        self.model.save_model(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.features_seleccionados, self.features_path)