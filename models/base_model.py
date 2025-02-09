from abc import ABC, abstractmethod
from src.performance_monitor import PerformanceMonitor
from sklearn.model_selection import GridSearchCV
import joblib
import os
import time

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.y_pred = None
        self.y_pred_proba = None
        self.performance_monitor = PerformanceMonitor()
    
    @abstractmethod
    def create_model(self):
        pass
    
    @abstractmethod
    def get_param_grid(self):
        pass

    def train(self, X_train, y_train):
        cv = int(os.getenv('FRAUD_DETECTION_CROSS_VALIDATION_FOLDS', 3))
        
        self.performance_monitor.start_monitoring(self.name, phase='training')
        param_grid = self.get_param_grid()
        print(f"Parâmetros: {param_grid}")
        
        print("\nIniciando GridSearch...")
        start_time = time.time()
        grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',
                verbose=3,
                n_jobs=1
            )
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"\nBusca concluída em {end_time - start_time:.2f} segundos")
        self.performance_monitor.stop_monitoring(self.name, phase='training')
        self.model = grid_search.best_estimator_
        print(f"\nMelhores parâmetros: {grid_search.best_params_}")

        return grid_search

    
    def predict(self, X_test):
        self.performance_monitor.start_monitoring(self.name, phase='prediction')
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        self.performance_monitor.stop_monitoring(self.name, phase='prediction')

        return self.y_pred, self.y_pred_proba
    
    def get_performance_metrics(self):
        return self.performance_monitor.metrics.get(self.name, {})
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = f"{path}/fraud_detection_model_{self.name}.joblib"
        joblib.dump(self.model, model_path)
        print(f"\nModelo {self.name} salvo como '{model_path}'")
