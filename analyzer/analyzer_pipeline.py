from typing import Type, Any

import pandas as pd

from analyzer.base_analyzer import BaseAnalyzer


class AnalyzerPipeline:
    def __init__(self, analyzer_classes: list[Type[BaseAnalyzer]]):
        self.pipeline = list()
        for analyzer_class in analyzer_classes:
            self.pipeline.append(analyzer_class())

    def run(self, data: pd.DataFrame) -> dict[str, Any]:
        context = dict()
        for analyzer in self.pipeline:
            new_context = analyzer.analyze(data, context)
            context.update(new_context)
        return context
