from src.model_trainer import ModelTrainer
from src.model_tester import ModelTester
from src.model_evaluator import ModelEvaluator
from models.knn import KNNModel
from models.xgboost import XGBoostModel
from models.mlp import MLPModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.svm import SVMModel
import os
import numpy as np

class ModelOperationsHandler:
    def __init__(self, discord_logger, desired_models=None, only_train_models=False, only_test_models=False, output_dir=None):
        self.discord_logger = discord_logger
        self.models = self._default_models() if desired_models is None else desired_models
        self.trainer = ModelTrainer(self.models, self.discord_logger)
        self.tester = ModelTester(self.models, self.discord_logger)
        self.evaluator = ModelEvaluator(f'{output_dir}/images' if output_dir else 'output/images')
        self.only_train_models = only_train_models
        self.only_test_models = only_test_models
        self.results = {'training': None, 'testing': None, 'evaluation': None}
        self.metrics_dir = f'{output_dir}/metrics' if output_dir else 'output/metrics'
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

    def _default_models(self):
        return [
            KNNModel(),
            XGBoostModel(),
            MLPModel(),
            RandomForestModel(),
            LogisticRegressionModel(),
            SVMModel()
        ]

    def execute_operations(self, X_train=None, y_train=None, X_test=None, y_test=None, feature_columns=None):
        best_model = None
        
        if self.only_train_models:
            print("\n=== Executando apenas treinamento ===")
            self.results['training'] = self.trainer.train_models(X_train, y_train)
            self._print_training_metrics()
            self._save_metrics('training')
            best_model = self._determine_best_model_from_training()
        
        elif self.only_test_models:
            print("\n=== Executando apenas teste ===")
            self.results['testing'] = self.tester.test_models(X_test)
            if y_test is not None and self.results['testing']:
                self._evaluate_and_plot_results(y_test, feature_columns)
                self._print_test_metrics(y_test)
                self._save_metrics('testing')
            
        else:
            print("\n=== Executando treinamento e teste ===")
            self.results['training'] = self.trainer.train_models(X_train, y_train)
            self.results['testing'] = self.tester.test_models(X_test)
            
            self._print_training_metrics()
            
            if y_test is not None and self.results['testing']:
                self._evaluate_and_plot_results(y_test, feature_columns)
                self._print_test_metrics(y_test)
                self._save_metrics('all')
            
            best_model = self._determine_best_model_from_training()
            
        return best_model

    def _determine_best_model_from_training(self):
        best_model = None
        best_score = -1
        if self.results['training']:
            for model_name, result in self.results['training'].items():
                if result['best_score'] > best_score:
                    best_score = result['best_score']
                    best_model = next((model for model in self.models if model.name == model_name), None)
        return best_model

    def _print_training_metrics(self):
        if not self.results['training']:
            return
            
        print("\n" + "="*50)
        print("             MÉTRICAS DE TREINAMENTO")
        print("="*50)
        
        for model_name, result in self.results['training'].items():
            print(f"\n{'-'*20} {model_name} {'-'*20}")
            print(f"Melhor Score (F1): {result['best_score']:.4f}")
            
            print("\nMelhores Parâmetros:")
            for param, value in result['best_params'].items():
                print(f"  - {param}: {value}")
                
            if 'performance_metrics' in result and result['performance_metrics']:
                metrics = result['performance_metrics']
                print("\nMétricas de Performance:")
                
                if 'training' in metrics and metrics['training']:
                    train_metrics = metrics['training']
                    print("  Treinamento Total:")
                    print(f"    - Tempo de Execução: {train_metrics.get('execution_time', 0):.2f} segundos")
                    print(f"    - Uso de CPU: {train_metrics.get('cpu_usage', 0):.2f}%")
                    print(f"    - Uso de Memória: {train_metrics.get('memory_usage', 0):.2f} MB")
                
                if 'grid_search' in metrics and metrics['grid_search']:
                    grid_metrics = metrics['grid_search']
                    print("\n  Grid Search:")
                    print(f"    - Tempo de Execução: {grid_metrics.get('execution_time', 0):.2f} segundos")
                    print(f"    - Uso de CPU: {grid_metrics.get('cpu_usage', 0):.2f}%")
                    print(f"    - Uso de Memória: {grid_metrics.get('memory_usage', 0):.2f} MB")
            print("-"*50)

    def _print_test_metrics(self, y_test):
        if not self.results['testing'] or not self.results['evaluation']:
            return
            
        print("\n" + "="*50)
        print("               MÉTRICAS DE TESTE")
        print("="*50)
        
        print("\nSumário Comparativo:")
        print("-"*50)
        self.evaluator.print_metrics_summary(self.results['evaluation'])
        
        for model_name, result in self.results['testing'].items():
            if result['status'] == 'Success':
                print(f"\n{'-'*20} {model_name} {'-'*20}")
                
                if 'performance_metrics' in result and result['performance_metrics']:
                    metrics = result['performance_metrics']
                    print("\nMétricas de Performance:")
                    
                    if 'testing' in metrics and metrics['testing']:
                        test_metrics = metrics['testing']
                        print("  Teste Total:")
                        print(f"    - Tempo de Execução: {test_metrics.get('execution_time', 0):.2f} segundos")
                        print(f"    - Uso de CPU: {test_metrics.get('cpu_usage', 0):.2f}%")
                        print(f"    - Uso de Memória: {test_metrics.get('memory_usage', 0):.2f} MB")
                    
                    if 'prediction' in metrics and metrics['prediction']:
                        pred_metrics = metrics['prediction']
                        print("\n  Predição:")
                        print(f"    - Tempo de Execução: {pred_metrics.get('execution_time', 0):.2f} segundos")
                        print(f"    - Uso de CPU: {pred_metrics.get('cpu_usage', 0):.2f}%")
                        print(f"    - Uso de Memória: {pred_metrics.get('memory_usage', 0):.2f} MB")
                
                print("\nRelatório de Classificação:")
                self.evaluator.print_classification_report(y_test, result['y_pred'], model_name)
                
                if model_name in self.results['evaluation']:
                    eval_metrics = self.results['evaluation'][model_name]
                    print("\nMétricas Principais:")
                    print(f"  - Accuracy:  {eval_metrics['accuracy']:.4f}")
                    print(f"  - Precision: {eval_metrics['precision']:.4f}")
                    print(f"  - Recall:    {eval_metrics['recall']:.4f}")
                    print(f"  - F1-Score:  {eval_metrics['f1_score']:.4f}")
                    print(f"  - ROC AUC:   {eval_metrics['roc_auc']:.4f}")
                print("-"*50)

    def _save_metrics(self, phase='all'):
        import json
        from datetime import datetime
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if phase in ['all', 'training'] and self.results['training']:
            training_metrics = convert_to_serializable(self.results['training'])
            training_file = os.path.join(self.metrics_dir, f'training_metrics_{timestamp}.json')
            with open(training_file, 'w') as f:
                json.dump(training_metrics, f, indent=4)
            print(f"\nMétricas de treinamento salvas em: {training_file}")
            
        if phase in ['all', 'testing'] and self.results['testing'] and self.results['evaluation']:
            combined_metrics = {
                'predictions': convert_to_serializable(self.results['testing']),
                'evaluation': convert_to_serializable(self.results['evaluation'])
            }
            testing_file = os.path.join(self.metrics_dir, f'testing_metrics_{timestamp}.json')
            with open(testing_file, 'w') as f:
                json.dump(combined_metrics, f, indent=4)
            print(f"Métricas de teste salvas em: {testing_file}")

    def _evaluate_and_plot_results(self, y_test, feature_columns):
        self.results['evaluation'] = self.evaluator.evaluate_all_models(self.results['testing'], y_test)
        
        for model_name, result in self.results['testing'].items():
            if result['status'] == 'Success':
                self.evaluator.plot_confusion_matrix(y_test, result['y_pred'], model_name)
                model = next((m for m in self.models if m.name == model_name), None)
                if model and hasattr(model.model, 'feature_importances_'):
                    self.evaluator.plot_feature_importance(model.model, feature_columns, model_name)
