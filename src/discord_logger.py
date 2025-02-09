import requests
import json
from datetime import datetime
import os
from typing import Optional

class DiscordLogger:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None
        
    def send_message(self, content, title=None, color=0x00ff00):
        """Envia uma mensagem para o Discord"""
        if not self.enabled:
            return
            
        data = {
            "embeds": [{
                "description": content,
                "color": color
            }]
        }
        
        if title:
            data["embeds"][0]["title"] = title
            
        try:
            response = requests.post(self.webhook_url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Erro ao enviar mensagem para o Discord: {str(e)}")
            
    def send_file(self, file_path, content=None):
        """Envia um arquivo para o Discord"""
        if not self.enabled:
            return
            
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {}
                if content:
                    data['content'] = content
                    
                response = requests.post(self.webhook_url, files=files, data=data)
                response.raise_for_status()
        except Exception as e:
            print(f"Erro ao enviar arquivo para o Discord: {str(e)}")
            
    def send_training_start(self, dataset_info):
        """Notifica início do treinamento"""
        if not self.enabled:
            return
            
        message = (
            "🚀 **Iniciando Processo de Detecção de Fraudes**\n\n"
            f"📊 **Informações do Dataset:**\n"
            f"```\n{dataset_info}\n```\n"
            f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message, title="Início do Treinamento", color=0x3498db)
        
    def send_model_metrics(self, model_name, metrics):
        """Envia métricas de um modelo específico"""
        if not self.enabled:
            return
            
        message = (
            f"📈 **Métricas do Modelo {model_name}**\n\n"
            f"Acurácia: {metrics['accuracy']:.4f}\n"
            f"Precisão: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"F1-Score: {metrics['f1_score']:.4f}\n"
            f"ROC AUC: {metrics['roc_auc']:.4f}"
        )
        self.send_message(message, title=f"Avaliação - {model_name}", color=0x2ecc71)
        
    def send_performance_metrics(self, model_name, metrics):
        """Envia métricas de performance de um modelo"""
        if not self.enabled:
            return
            
        message = (
            f"⚡ **Performance do Modelo {model_name}**\n\n"
            f"**Tempo de Execução:**\n"
            f"Treino: {metrics.get('training_time', 0):.2f}s\n"
            f"Predição: {metrics.get('prediction_time', 0):.2f}s\n\n"
            f"**Uso de CPU:**\n"
            f"Treino: {metrics.get('training_cpu', 0):.1f}%\n"
            f"Predição: {metrics.get('prediction_cpu', 0):.1f}%\n\n"
            f"**Uso de Memória:**\n"
            f"Treino: {metrics.get('training_memory', 0):.1f}MB\n"
            f"Predição: {metrics.get('prediction_memory', 0):.1f}MB"
        )
        self.send_message(message, title=f"Performance - {model_name}", color=0xe74c3c)
        
    def send_training_complete(self, best_model, total_time):
        """Notifica conclusão do treinamento"""
        if not self.enabled:
            return
            
        message = (
            "✅ **Processo Finalizado!**\n\n"
            f"🏆 Melhor Modelo: {best_model}\n"
            f"⏱️ Tempo Total: {total_time:.2f} segundos\n\n"
            "📁 Modelos salvos e visualizações geradas."
        )
        self.send_message(message, title="Treinamento Concluído", color=0x9b59b6)
        
    def send_plots(self):
        """Envia os plots gerados"""
        if not self.enabled:
            return
            
        plot_files = [
            ('distribuicao_classes.png', 'Distribuição das Classes'),
            ('distribuicao_valores.png', 'Distribuição dos Valores'),
            ('correlacao_features.png', 'Correlação entre Features'),
            ('comparacao_metricas.png', 'Comparação de Métricas'),
            ('comparacao_roc_curves.png', 'Curvas ROC'),
            ('comparacao_pr_curves.png', 'Curvas Precision-Recall')
        ]
        
        for filename, description in plot_files:
            filepath = os.path.join('output/images', filename)
            if os.path.exists(filepath):
                self.send_file(filepath, f"📊 {description}")
                
    def send_confusion_matrices(self):
        """Envia as matrizes de confusão"""
        if not self.enabled:
            return
            
        for filename in os.listdir('output/images'):
            if filename.startswith('matriz_confusao_'):
                filepath = os.path.join('output/images', filename)
                model_name = filename.replace('matriz_confusao_', '').replace('.png', '')
                self.send_file(filepath, f"🎯 Matriz de Confusão - {model_name}")
                
    def send_feature_importance(self):
        """Envia os gráficos de importância de features"""
        if not self.enabled:
            return
            
        for filename in os.listdir('output/images'):
            if filename.startswith('feature_importance_'):
                filepath = os.path.join('output/images', filename)
                model_name = filename.replace('feature_importance_', '').replace('.png', '')
                self.send_file(filepath, f"🔍 Importância das Features - {model_name}") 