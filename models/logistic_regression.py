from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
import numpy as np

class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("LogisticRegression")
        self.random_state = random_state
        self.create_model()
    
    def create_model(self):
        self.model = LogisticRegression(
            random_state=self.random_state,
            solver='saga',
            max_iter=1000,
            C=0.1,
            penalty='l2',
            tol=1e-3,
            n_jobs=-1,
            warm_start=True,
            class_weight='balanced'
        )

    def predict(self, X_test):
        self.performance_monitor.start_monitoring(self.name, phase='prediction')
        
        batch_size = 5000
        total_samples = len(X_test)
        
        y_pred = np.zeros(total_samples)
        y_pred_proba = np.zeros(total_samples)
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch = X_test[i:end_idx]
            
            y_pred[i:end_idx] = self.model.predict(batch)
            y_pred_proba[i:end_idx] = self.model.predict_proba(batch)[:, 1]
        
        self.performance_monitor.stop_monitoring(self.name, phase='prediction')
        return y_pred, y_pred_proba
    
    def get_param_grid(self):
        return {
            'C': [0.1, 1.0],
            'penalty': ['l2'],
            'solver': ['saga'],
            'class_weight': ['balanced']
        }
