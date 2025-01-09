import matplotlib.pyplot as plt
import seaborn as sns

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
