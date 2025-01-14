import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_risk_by_life_stage(data, model, selected_features, target_columns):
    """
    생애 주기별 위험도를 시각화
    """
    stages = data['Life_Stage'].unique()
    risk_scores = {target: [] for target in target_columns}

    for stage in stages:
        stage_data = data[data['Life_Stage'] == stage][selected_features]
        predictions = model.predict_proba(stage_data)

        for i, target in enumerate(target_columns):
            risk_scores[target].append(predictions[i][:, 1].mean())  # 평균 위험도

    # 시각화
    for target in target_columns:
        sns.barplot(x=stages, y=risk_scores[target])
        plt.title(f"{target} Risk by Life Stage")
        plt.xlabel("Life Stage")
        plt.ylabel("Risk Probability")
        plt.show()

def save_results(results, file_name):
    """
    모델 테스트 결과를 CSV 파일로 저장합니다.

    Args:
        results (list or dict): 저장할 결과 데이터 (모델 이름과 성능 지표들).
        file_name (str): 저장할 파일 경로.

    Returns:
        None
    """
    # 결과 데이터를 pandas DataFrame으로 변환
    if isinstance(results, list):
        results_df = pd.DataFrame(results)
    elif isinstance(results, dict):
        results_df = pd.DataFrame([results])
    else:
        raise ValueError("Results must be a list or a dictionary.")

    # CSV 파일로 저장
    results_df.to_csv(file_name, index=False, encoding='utf-8')
    print(f"Results saved to {file_name}")

