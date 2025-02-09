from src.performance_monitor import PerformanceMonitor
import os
import joblib

class ModelTester:
    def __init__(self, models, discord_logger):
        self.models = models
        self.results = {}
        self.performance_monitor = PerformanceMonitor()
        self.discord_logger = discord_logger
    
    def load_models(self):
        models_dir = 'output/models'
        for model in self.models:
            model_path = os.path.join(models_dir, f'fraud_detection_model_{model.name}.joblib')
            if os.path.exists(model_path):
                model.model = joblib.load(model_path)
                self.results[model.name] = 'Model loaded successfully'
            else:
                self.results[model.name] = 'Model not found'
 
    def test_models(self, X_test):
        self.load_models()
        testing_results = {}
        
        for model in self.models:
            if model.model is None:
                testing_results[model.name] = {
                    'status': 'Model not loaded',
                    'y_pred': None,
                    'y_pred_proba': None,
                    'performance_metrics': None
                }
                
                if self.discord_logger:
                    message = f"❌ Erro ao testar {model.name}: Modelo não encontrado"
                    self.discord_logger.send_message(message)
                    
                continue
                
            self.performance_monitor.start_monitoring(model.name, phase='testing')
            
            model.performance_monitor.start_monitoring(model.name, phase='prediction')
            y_pred, y_pred_proba = model.predict(X_test)
            model.performance_monitor.stop_monitoring(model.name, phase='prediction')
            
            self.performance_monitor.stop_monitoring(model.name, phase='testing')
            
            model_metrics = model.get_performance_metrics()
            test_metrics = self.performance_monitor.metrics.get(model.name, {})
            
            prediction_metrics = model_metrics.get('prediction', {})
            testing_metrics = test_metrics.get('testing', {})
            
            if prediction_metrics and testing_metrics:
                testing_metrics['cpu_usage'] = max(testing_metrics.get('cpu_usage', 0), 
                                                 prediction_metrics.get('cpu_usage', 0))
                testing_metrics['memory_usage'] = max(testing_metrics.get('memory_usage', 0), 
                                                    prediction_metrics.get('memory_usage', 0))
                testing_metrics['execution_time'] = max(testing_metrics.get('execution_time', 0), 
                                                      prediction_metrics.get('execution_time', 0))
            
            testing_results[model.name] = {
                'status': 'Success',
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'performance_metrics': {
                    'testing': testing_metrics,
                    'prediction': prediction_metrics
                }
            }
            
            if self.discord_logger:
                message = (
                    f"✅ Teste do modelo {model.name} finalizado:\n"
                    f"- Tempo Total: {testing_metrics.get('execution_time', 0):.2f}s\n"
                    f"- CPU Média: {testing_metrics.get('cpu_usage', 0):.2f}%\n"
                    f"- Memória: {testing_metrics.get('memory_usage', 0):.2f}MB"
                )
                self.discord_logger.send_message(message)
        
        self.results = testing_results
        return self.results
