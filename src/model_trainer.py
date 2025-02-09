from src.performance_monitor import PerformanceMonitor
import os

class ModelTrainer:
    def __init__(self, models, discord_logger):
        self.models = models
        self.results = {}
        self.performance_monitor = PerformanceMonitor()
        self.discord_logger = discord_logger
    
    def train_models(self, X_train, y_train):
        training_results = {}
        for model in self.models:
            print(f"\nTreinando o modelo {model.name}...")
            
            self.performance_monitor.start_monitoring(model.name, phase='training')
            
            model.performance_monitor.start_monitoring(model.name, phase='grid_search')
            grid_search = model.train(X_train, y_train)
            model.performance_monitor.stop_monitoring(model.name, phase='grid_search')
            
            self.performance_monitor.stop_monitoring(model.name, phase='training')
            
            model_metrics = model.get_performance_metrics()
            train_metrics = self.performance_monitor.metrics.get(model.name, {})
            
            grid_search_metrics = model_metrics.get('grid_search', {})
            training_metrics = train_metrics.get('training', {})
            
            if grid_search_metrics and training_metrics:
                training_metrics['cpu_usage'] = max(training_metrics.get('cpu_usage', 0), 
                                                  grid_search_metrics.get('cpu_usage', 0))
                training_metrics['memory_usage'] = max(training_metrics.get('memory_usage', 0), 
                                                     grid_search_metrics.get('memory_usage', 0))
                training_metrics['execution_time'] = max(training_metrics.get('execution_time', 0), 
                                                       grid_search_metrics.get('execution_time', 0))
            
            training_results[model.name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'performance_metrics': {
                    'training': training_metrics,
                    'grid_search': grid_search_metrics
                }
            }
            self.save_model(model)

            if self.discord_logger:
                message = (
                    f"🎯 Treinamento do modelo {model.name} finalizado:\n"
                    f"- Melhor F1-Score: {grid_search.best_score_:.4f}\n"
                    f"- Tempo Total: {training_metrics.get('execution_time', 0):.2f}s\n"
                    f"- CPU Média: {training_metrics.get('cpu_usage', 0):.2f}%\n"
                    f"- Memória: {training_metrics.get('memory_usage', 0):.2f}MB"
                )
                self.discord_logger.send_message(message)

        self.results = training_results
        return self.results

    def save_model(self, model, base_path='output/models'):
        os.makedirs(base_path, exist_ok=True)
        model.save_model(base_path)
