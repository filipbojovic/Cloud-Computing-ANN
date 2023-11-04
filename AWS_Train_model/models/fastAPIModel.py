from pydantic import BaseModel

class ModelAPI(BaseModel):
    model_id: int
    model_name: str
    dataset_name: str
    mse: float
    auc: float
    acc: float