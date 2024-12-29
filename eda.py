import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    """
    탐색적 데이터 분석 수행
    """
    # Age 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(data[' Age (yrs)'], kde=True)
    plt.title('Age Distribution')
    plt.show()

    # Boxplot (Age와 Target 관계)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Target'], y=data[' Age (yrs)'])
    plt.title('Age vs Target')
    plt.show()

    # 주요 변수 상관관계
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Target 별 주요 변수 밀도 비교
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data[data['Target'] == 0]['BMI'], label='Target=0', shade=True)
    sns.kdeplot(data[data['Target'] == 1]['BMI'], label='Target=1', shade=True)
    plt.title('BMI Distribution by Target')
    plt.legend()
    plt.show()
