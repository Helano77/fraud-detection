from sklearn.neural_network import MLPClassifier
from models.base_model import BaseModel
import numpy as np

class MLPModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("NeuralNetwork")
        self.random_state = random_state
        self.create_model()
    
    def create_model(self):
        # Arquitetura mais adequada para o tamanho do dataset
        # Input: 30 -> Hidden: 128, 64 -> Output: 2
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size=256,  # Batch size fixo para melhor controle de memória
            learning_rate_init=0.001,  # Taxa de aprendizado inicial fixa
            learning_rate='constant',  # Taxa constante para melhor previsibilidade
            max_iter=200,  # Número máximo de épocas
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
            tol=1e-4
        )
        
    def predict(self, X_test):
        self.performance_monitor.start_monitoring(self.name, phase='prediction')
        
        batch_size = 1000
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
            'hidden_layer_sizes': [(64, 32), (128, 64)],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001],
            'batch_size': [256]
        }