import pandas as pd
from matplotlib import pyplot as plt

from analyzer.base_analyzer import BaseAnalyzer


class ContinuousBoxAnalyzer(BaseAnalyzer):
    def analyze(self, data: pd.DataFrame, params: dict) -> dict:
        target_columns = [
            'CreditScore',
            'OriginalCLTV',
            'OriginalDTI',
            'OriginalUPB',
            'OriginalLTV',
            'OriginalInterestRate',
        ]

        cols = 3
        rows = len(target_columns) // cols + (len(target_columns) % cols > 0)

        # 创建画板
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # 如果只有一行，将axes转换为二维数组以便统一处理
        if rows == 1:
            axes = [axes]

        # 展平axes数组以便迭代
        axes_flat = axes.flatten() if rows > 1 else axes

        # 为每个目标列创建箱线图
        for i, column in enumerate(target_columns):
            if column in data.columns:
                # 获取数据并计算null值数量
                column_data = data[column]
                null_count = column_data.isnull().sum()

                # 绘制箱线图
                axes_flat[i].boxplot(column_data.dropna(), patch_artist=True,
                                   boxprops=dict(facecolor='lightblue', color='blue'),
                                   medianprops=dict(color='red'))

                # 设置标题和标签
                axes_flat[i].set_title(f'{column} Box Plot')
                axes_flat[i].set_ylabel('Value')

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
