from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analyzer.base_analyzer import BaseAnalyzer


class DiscreteDistributionAnalyzer(BaseAnalyzer):
    def analyze(self, data: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
        target_columns = [
            'target',
            'FirstPaymentDate',
            'FirstTimeHomebuyerFlag',
            'MaturityDate',
            'NumberOfUnits',
            'OccupancyStatus',
            'Channel',
            'PPM_Flag',
            'ProductType',
            'PropertyState',
            'PropertyType',
            'LoanPurpose',
            'OriginalLoanTerm',
            'NumberOfBorrowers',
            'SellerName',
            'ServicerName',
            'SuperConformingFlag',
            'PreHARP_Flag',
            'ProgramIndicator',
            'ReliefRefinanceIndicator',
            'PropertyValMethod',
            'InterestOnlyFlag',
            'BalloonIndicator',
        ]

        cols = 3
        rows = int(np.ceil(len(target_columns) / cols))

        # 创建画板
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # 如果只有一行，将axes转换为二维数组以便统一处理
        if rows == 1:
            axes = [axes]

        # 展平axes数组以便迭代
        axes_flat = axes.flatten() if rows > 1 else axes

        # 为每个目标列创建分布图
        for i, column in enumerate(target_columns):
            if column in data.columns:
                # 获取数据并计算null值数量
                column_data = data[column]
                null_count = column_data.isnull().sum()

                # 计算值计数（适用于字符串和数字）
                value_counts = column_data.value_counts(dropna=True)

                # 如果类别太多，只显示前20个
                if len(value_counts) > 20:
                    value_counts = value_counts[:20]
                    # 添加省略标记
                    others_count = len(column_data.dropna()) - sum(value_counts)
                    if others_count > 0:
                        value_counts['Others'] = others_count

                # 绘制条形图
                axes_flat[i].bar(range(len(value_counts)), value_counts.values,
                                 color='lightcoral', edgecolor='black')

                # 设置x轴标签
                axes_flat[i].set_xticks(range(len(value_counts)))
                axes_flat[i].set_xticklabels(value_counts.index, rotation=45, ha='right')

                # 设置标题和标签
                axes_flat[i].set_title(f'{column} Distribution')
                axes_flat[i].set_xlabel(column)
                axes_flat[i].set_ylabel('Count')

                # 在右上角添加null值数量
                axes_flat[i].text(0.95, 0.95, f'null count={null_count}',
                                  transform=axes_flat[i].transAxes,
                                  verticalalignment='top',
                                  horizontalalignment='right',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 隐藏多余的子图
        for i in range(len(target_columns), len(axes_flat)):
            axes_flat[i].set_visible(False)

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.show()

        return {}
