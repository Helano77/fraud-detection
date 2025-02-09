import os
from typing import Optional
from dotenv import load_dotenv
from models.knn import KNNModel
from models.xgboost import XGBoostModel
from models.mlp import MLPModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.svm import SVMModel

class Config:
    """Configurações da aplicação baseadas em variáveis de ambiente"""
    
    # Carrega as variáveis de ambiente do arquivo .env.fraud
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.fraud')
    if os.path.exists(env_file):
        load_dotenv(env_file)
    
    @staticmethod
    def get_discord_webhook() -> Optional[str]:
        """Retorna a URL do webhook do Discord ou None se não configurado"""
        return os.getenv('FRAUD_DETECTION_DISCORD_WEBHOOK')
    
    @staticmethod
    def is_logging_enabled() -> bool:
        """Retorna True se o logging está ativado, False caso contrário"""
        return os.getenv('FRAUD_DETECTION_LOGGING_ENABLED', 'false').lower() == 'true'
    
    @staticmethod
    def validate_config() -> bool:
        """Valida se as configurações necessárias estão presentes quando logging está ativado"""
        if Config.is_logging_enabled():
            webhook_url = Config.get_discord_webhook()
            if not webhook_url:
                print("AVISO: Logging está ativado mas FRAUD_DETECTION_DISCORD_WEBHOOK não está configurado")
                return False
            print("Logging no Discord está ativado e configurado corretamente")
        else:
            print("Logging no Discord está desativado")
        return True 
    
    @staticmethod
    def get_desired_models() -> list[str]:
        """Retorna a lista de modelos desejados"""
        model_names_array = os.getenv('FRAUD_DETECTION_DESIRED_MODELS', '').split(',')
        if model_names_array == ['']:
            return None
        
        models_map = {
            'KNN': KNNModel(),
            'XGBoost': XGBoostModel(),
            'MLP': MLPModel(),
            'RandomForest': RandomForestModel(),
            'LogisticRegression': LogisticRegressionModel(),
            'SVM': SVMModel()
        }
        
        models = [models_map[model_name] for model_name in model_names_array]

        return models
    
    @staticmethod
    def get_only_train_models() -> bool:
        """Retorna True se apenas os modelos de treinamento estão ativos, False caso contrário"""
        return os.getenv('FRAUD_DETECTION_ONLY_TRAIN_MODELS', 'false').lower() == 'true'
    
    @staticmethod
    def get_only_test_models() -> bool:
        """Retorna True se apenas os modelos de teste estão ativos, False caso contrário"""
        return os.getenv('FRAUD_DETECTION_ONLY_TEST_MODELS', 'false').lower() == 'true'
    
    