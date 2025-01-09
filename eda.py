import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    """
    탐색적 데이터 분석 수행
    """
    # Age 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age (yrs)'], kde=True)
    plt.title('Age Distribution')
    plt.show()

    # Boxplot (Age와 Target 관계)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Target'], y=data['Age (yrs)'])
    plt.title('Age vs Target')
    plt.show()

def analyze_by_life_stage(data, variables):
    """
    생애 주기별 주요 변수 분포 분석
    """
    stages = data['Life_Stage'].unique()
    for var in variables:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Life_Stage', y=var, data=data)
        plt.title(f'{var} Distribution by Life Stage')
        plt.show()
