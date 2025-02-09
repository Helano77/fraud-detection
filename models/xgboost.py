from xgboost import XGBClassifier
from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.create_model()
    
    def create_model(self):
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def get_param_grid(self):
        return {
            'n_estimators': [100, 200],
            'max_depth': [6, 8],
            'learning_rate': [0.1],
            'scale_pos_weight': [1]
        }
