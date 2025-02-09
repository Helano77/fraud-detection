import seaborn as sns
import matplotlib.pyplot as plt
from src.data_preprocessor import DataPreprocessor
from src.model_operations_handler import ModelOperationsHandler
from src.discord_logger import DiscordLogger
from src.config import Config
import time
import os
from datetime import datetime

def setup_config():
    Config.validate_config()
    logging_enabled = Config.is_logging_enabled()
    discord_webhook = None if not logging_enabled else Config.get_discord_webhook()
    desired_models = None if Config.get_desired_models() == [''] else Config.get_desired_models()
    only_train_models = Config.get_only_train_models()
    only_test_models = Config.get_only_test_models()

    return discord_webhook, desired_models, only_train_models, only_test_models

def create_output_dirs(timestamp):
    base_dir = f'output/{timestamp}'
    subdirs = ['images', 'metrics']
    
    for subdir in subdirs:
        os.makedirs(f'{base_dir}/{subdir}', exist_ok=True)
    
    os.makedirs('output/models', exist_ok=True)
    
    return base_dir

def main():
    # Configuração inicial
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = create_output_dirs(timestamp)
    
    discord_webhook, desired_models, only_train_models, only_test_models = setup_config()
    discord_logger = DiscordLogger(discord_webhook)
    
    preprocessor = DataPreprocessor('data/creditcard.csv', f'{output_dir}/images')

    start_time = time.time()
    
    # Carregar e pré-processar os dados
    print("Iniciando o processo de detecção de fraudes...")
    df = preprocessor.load_data()
    preprocessor.split_data()
    preprocessor.scale_features()

    # Logging de informações iniciais
    dataset_info = (
        f"Total de amostras: {len(df)}\n"
        f"Número de features: {len(df.columns) - 1}\n"
        f"Distribuição das classes:\n{df['Class'].value_counts().to_string()}"
    )
    discord_logger.send_training_start(dataset_info)
    
    # Plotar visualizações iniciais dos dados
    print("\nGerando visualizações iniciais dos dados...")
    preprocessor.plot_class_distribution(df)
    preprocessor.plot_amount_distribution(df)
    preprocessor.plot_correlation_matrix(preprocessor.X)
    
    # Enviar plots iniciais
    discord_logger.send_plots()
    
    # Criar e executar o handler de operações dos modelos
    operations_handler = ModelOperationsHandler(
        discord_logger=discord_logger,
        desired_models=desired_models,
        only_train_models=only_train_models,
        only_test_models=only_test_models,
        output_dir=output_dir
    )
    
    # Executar operações de treino e/ou teste
    best_model = operations_handler.execute_operations(
        X_train=preprocessor.X_train_scaled,
        y_train=preprocessor.y_train,
        X_test=preprocessor.X_test_scaled,
        y_test=preprocessor.y_test,
        feature_columns=preprocessor.X.columns
    )    

    # Calcular tempo total
    total_time = time.time() - start_time
    
    # Enviar mensagem de conclusão
    discord_logger.send_training_complete(best_model, total_time)
    
    print(f"\nProcesso finalizado! Todos os resultados foram salvos em: {output_dir}")

if __name__ == "__main__":
    main()
