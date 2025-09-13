from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseAnalyzer(ABC):

    @abstractmethod
    def analyze(self, data: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
        pass
