from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def analyze_life_stage(data, selected_features):
    """
    생애 주기별 데이터 분석 및 모델 평가
    """
    stages = data['Life_Stage'].unique()
    for stage in stages:
        print(f"Analyzing Life Stage: {stage}")
        stage_data = data[data['Life_Stage'] == stage]
        X_stage = stage_data[selected_features]
        y_stage = stage_data['Target']

        # 데이터 분할 및 모델 훈련
        X_train, X_test, y_train, y_test = train_test_split(X_stage, y_stage, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # 평가
        y_pred = rf_model.predict(X_test)
        print(f"Accuracy for {stage}:", accuracy_score(y_test, y_pred))
