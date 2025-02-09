from sklearn.svm import SVC
from models.base_model import BaseModel
import pandas as pd
import numpy as np

class SVMModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__("SVM")
        self.random_state = random_state
        self.create_model()

    def train(self, X_train, y_train):
        X_processed, y_processed = self.preprocess_data(X_train, y_train)
        return super().train(X_processed, y_processed)
    
    def create_model(self):
        self.model = SVC(
            random_state=self.random_state,
            probability=True,
            cache_size=2000,
            class_weight='balanced'
        )

    def preprocess_data(self, X_train, y_train):
        if len(X_train) > 100000:
            df = pd.DataFrame(X_train)
            df['target'] = y_train
            
            class_counts = df['target'].value_counts()
            target_size = min(10000, class_counts.min())
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df['target'] == class_label]
                sampled_df = class_df.sample(
                    n=target_size,
                    random_state=self.random_state,
                    replace=False
                )
                balanced_dfs.append(sampled_df)
            
            df = pd.concat(balanced_dfs, axis=0).reset_index(drop=True)
            X_train = df.drop('target', axis=1).values
            y_train = df['target'].values
            
        return X_train, y_train
    
    def get_param_grid(self):
        return {
            'C': [1.0],
            'kernel': ['rbf'],
            'gamma': ['scale'],
            'class_weight': ['balanced']
        }