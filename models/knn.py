from sklearn.neighbors import KNeighborsClassifier
from models.base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors
        self.create_model()
    
    def create_model(self):
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors
        )
    
    def get_param_grid(self):
        return {
            'n_neighbors': [5, 7],
            'weights': ['distance'],
            'metric': ['euclidean'],
            'algorithm': ['auto']
        }