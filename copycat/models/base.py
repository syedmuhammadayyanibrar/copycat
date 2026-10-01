from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseCreatorModel(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseCreatorModel":
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        pass
