import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
class DataPreprocessor:
    def __init__(self, data_path, visualization_dir):
        self.data_path = data_path
        self.visualization_dir = visualization_dir
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        print("Carregando o dataset...")
        df = pd.read_csv(self.data_path)
        df = df.drop('Time', axis=1)
        self.X = df.drop('Class', axis=1)
        self.y = df['Class']
        return df
    
    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
    def scale_features(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def plot_class_distribution(self, df):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Class')
        plt.title('Distribuição das Classes (0: Normal, 1: Fraude)')
        self._save_plot('distribuicao_classes.png')

    def plot_amount_distribution(self, df):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Class', y='Amount')
        plt.title('Distribuição dos Valores das Transações por Classe')
        self._save_plot('distribuicao_valores.png')

    def plot_correlation_matrix(self, X):
        plt.figure(figsize=(20, 16))
        sns.heatmap(X.corr(), cmap='coolwarm', center=0, annot=False)
        plt.title('Matriz de Correlação das Features')
        self._save_plot('correlacao_features.png')
    
    def _save_plot(self, filename):
        plt.savefig(os.path.join(self.visualization_dir, filename), bbox_inches='tight')
        plt.close()