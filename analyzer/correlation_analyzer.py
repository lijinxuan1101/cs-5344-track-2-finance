import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from analyzer.base_analyzer import BaseAnalyzer


class CorrelationAnalyzer(BaseAnalyzer):
    def analyze(self, data: pd.DataFrame, params: dict) -> dict:
        target_columns = [
            'CreditScore',
            'MI_Pct',
            'OriginalCLTV',
            'OriginalDTI',
            'OriginalUPB',
            'OriginalLTV',
            'OriginalInterestRate'
        ]

        # 选择目标列的数据
        corr_data = data[target_columns]

        # 计算相关性矩阵
        correlation_matrix = corr_data.corr()

        # 创建画板
        plt.figure(figsize=(10, 8))

        # 绘制热力图
        heatmap = sns.heatmap(correlation_matrix,
                              annot=True,  # 显示数值
                              cmap='coolwarm',  # 颜色映射
                              center=0,  # 颜色中心点
                              square=True,  # 正方形单元格
                              fmt='.2f',  # 数值格式
                              cbar=True)  # 显示颜色条

        # 设置标题和标签
        plt.title('Correlation Heatmap of Target Columns')
        plt.xlabel('Variables')
        plt.ylabel('Variables')

        # 旋转标签以提高可读性
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.show()

        return {}
