import pandas as pd

from analyzer.continuous_box_analyzer import ContinuousBoxAnalyzer
from analyzer.continuous_distribution_analyzer import ContinuousDistributionAnalyzer
from analyzer.correlation_analyzer import CorrelationAnalyzer
from analyzer.discrete_distribution_analyzer import DiscreteDistributionAnalyzer


def continuous_analyzer_test():
    data = pd.read_csv('data/raw_data/loans_train.csv')
    analyzer = ContinuousDistributionAnalyzer()
    result = analyzer.analyze(data, {})
    print(result)


def discrete_analyzer_test():
    data = pd.read_csv('data/raw_data/loans_train.csv')
    analyzer = DiscreteDistributionAnalyzer()
    result = analyzer.analyze(data, {})
    print(result)

def correlation_analyzer_test():
    data = pd.read_csv('data/raw_data/loans_train.csv')
    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(data, {})
    print(result)

def box_plot_analyzer_test():
    data = pd.read_csv('data/raw_data/loans_train.csv')
    analyzer = ContinuousBoxAnalyzer()
    result = analyzer.analyze(data, {})
    print(result)


if __name__ == '__main__':
    # continuous_analyzer_test()
    # discrete_analyzer_test()
    # correlation_analyzer_test()
    box_plot_analyzer_test()
