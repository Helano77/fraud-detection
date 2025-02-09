import time
import psutil
import pandas as pd
from datetime import datetime
import os
import threading
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        self.is_monitoring = False
        self.cpu_measurements = []
        self.monitor_thread = None
        
    def _cpu_monitor(self):
        """Thread dedicada para coletar medições de CPU"""
        while self.is_monitoring:
            measurements = [psutil.cpu_percent(interval=0.05) for _ in range(3)]
            self.cpu_measurements.append(np.mean(measurements))
            time.sleep(0.05)

    def start_monitoring(self, model_name, phase='training'):
        """Inicia o monitoramento de performance para uma fase específica"""
        import gc
        gc.collect()
        time.sleep(0.1)
        gc.collect()
        
        self.start_time = time.time()
        process = psutil.Process(os.getpid())

        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.cpu_measurements = []
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._cpu_monitor)
        self.monitor_thread.start()
        
    def stop_monitoring(self, model_name, phase='training'):
        """Para o monitoramento e registra as métricas para uma fase específica"""
        end_time = time.time()
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        import gc
        gc.collect()
        time.sleep(0.1)
        gc.collect()
        
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - self.start_time
        
        if self.cpu_measurements:
            cpu_array = np.array(self.cpu_measurements)
            mean_cpu = np.mean(cpu_array)
            std_cpu = np.std(cpu_array)
            filtered_cpu = cpu_array[abs(cpu_array - mean_cpu) <= 2 * std_cpu]
            cpu_usage = max(0, np.mean(filtered_cpu) if len(filtered_cpu) > 0 else mean_cpu)
        else:
            cpu_usage = 0
            
        memory_diff = max(0, end_memory - self.start_memory)
        
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
            
        self.metrics[model_name][phase] = {
            'execution_time': execution_time,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_diff
        }
        
    def save_metrics(self):
        if not os.path.exists('output'):
            os.makedirs('output')
            
        formatted_metrics = {}
        for model_name, phases in self.metrics.items():
            formatted_metrics[model_name] = {
                'training_time': phases.get('training', {}).get('execution_time', 0),
                'prediction_time': phases.get('prediction', {}).get('execution_time', 0),
                'training_cpu': phases.get('training', {}).get('cpu_usage', 0),
                'prediction_cpu': phases.get('prediction', {}).get('cpu_usage', 0),
                'training_memory': phases.get('training', {}).get('memory_usage', 0),
                'prediction_memory': phases.get('prediction', {}).get('memory_usage', 0)
            }
            
        df = pd.DataFrame.from_dict(formatted_metrics, orient='index')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/performance_metrics_{timestamp}.csv'
        df.to_csv(filename)
        print(f"\nMétricas de performance salvas em: {filename}")
        
    def print_summary(self):
        print("\n=== Métricas de Performance dos Modelos ===")
        
        print("\nTempo de Execução (segundos):")
        for model, phases in self.metrics.items():
            train_time = phases.get('training', {}).get('execution_time', 0)
            pred_time = phases.get('prediction', {}).get('execution_time', 0)
            print(f"{model:15} Treino: {train_time:.2f}s, Predição: {pred_time:.2f}s")
            
        print("\nUso Médio de CPU (percentual):")
        for model, phases in self.metrics.items():
            train_cpu = phases.get('training', {}).get('cpu_usage', 0)
            pred_cpu = phases.get('prediction', {}).get('cpu_usage', 0)
            print(f"{model:15} Treino: {train_cpu:.1f}%, Predição: {pred_cpu:.1f}%")
            
        print("\nUso Adicional de Memória (MB):")
        for model, phases in self.metrics.items():
            train_mem = phases.get('training', {}).get('memory_usage', 0)
            pred_mem = phases.get('prediction', {}).get('memory_usage', 0)
            print(f"{model:15} Treino: {train_mem:.1f}MB, Predição: {pred_mem:.1f}MB") 