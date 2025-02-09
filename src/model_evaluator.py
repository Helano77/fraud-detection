from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelEvaluator:
    def __init__(self, visualization_dir):
        self.visualization_dir = visualization_dir
        self._create_visualization_dir()
        
    def _create_visualization_dir(self):
        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)
        
    def evaluate_all_models(self, models_results, y_test):
        metrics = {}
        
        for model_name, result in models_results.items():
            y_pred = result['y_pred']
            y_pred_proba = result['y_pred_proba']
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics[model_name] = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'roc_auc': self._calculate_roc_auc(y_test, y_pred_proba)
            }
        
        self._plot_metrics_comparison(metrics)
        
        self._plot_multiple_roc_curves(models_results, y_test)
        
        self._plot_multiple_pr_curves(models_results, y_test)
        
        return metrics
    
    def _calculate_roc_auc(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return auc(fpr, tpr)
    
    def _plot_metrics_comparison(self, metrics):
        metrics_df = pd.DataFrame(metrics).T
        
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', width=0.8)
        plt.title('Comparação de Métricas entre Modelos')
        plt.xlabel('Modelos')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self._save_plot('comparacao_metricas.png')
        
    def _plot_multiple_roc_curves(self, models_results, y_test):
        plt.figure(figsize=(10, 8))
        
        for model_name, result in models_results.items():
            y_pred_proba = result['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curvas ROC - Comparação entre Modelos')
        plt.legend(loc="lower right")
        self._save_plot('comparacao_roc_curves.png')
        
    def _plot_multiple_pr_curves(self, models_results, y_test):
        plt.figure(figsize=(10, 8))
        
        for model_name, result in models_results.items():
            y_pred_proba = result['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name}')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall - Comparação entre Modelos')
        plt.legend(loc="lower left")
        self._save_plot('comparacao_pr_curves.png')
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name=""):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão {model_name}')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Previsto')
        self._save_plot(f'matriz_confusao_{model_name.lower()}.png')
        return cm
    
    def plot_feature_importance(self, model, feature_names, model_name=""):
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
            plt.title(f'15 Features Mais Importantes - {model_name}')
            self._save_plot(f'feature_importance_{model_name.lower()}.png')
            return feature_importance
        return None
    
    def _save_plot(self, filename):
        plt.savefig(os.path.join(self.visualization_dir, filename), bbox_inches='tight')
        plt.close()
        
    def print_classification_report(self, y_true, y_pred, model_name=""):
        print(f"\nRelatório de Classificação - {model_name}:")
        print(classification_report(y_true, y_pred))
        
    def print_metrics_summary(self, metrics):
        print("\nSumário das métricas principais:")
        metrics_df = pd.DataFrame(metrics).T
        print(metrics_df.round(4))

