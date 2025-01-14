def perform_eda(data):
    """
    탐색적 데이터 분석 수행.
    """
    print("데이터 요약:")
    print(data.describe())

def analyze_by_life_stage(data, variables):
    """
    생애 주기별 데이터 분석.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    for var in variables:
        sns.boxplot(x='Life_Stage', y=var, data=data)
        plt.title(f'{var} Distribution by Life Stage')
        plt.show()
