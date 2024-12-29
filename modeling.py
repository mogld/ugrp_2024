from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_and_evaluate_model(data):
    """
    모델 훈련 및 평가
    """
    # 독립 변수(X)와 종속 변수(y) 분리
    selected_features = [' Age (yrs)', 'BMI', 'AMH(ng/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'RBS(mg/dl)', 'TSH (mIU/L)', 'PCOS (Y/N)']
    if not all(feature in data.columns for feature in selected_features):
        missing_features = [feature for feature in selected_features if feature not in data.columns]
        raise ValueError(f"The following required features are missing in the dataset: {missing_features}")

    X = data[selected_features]
    y = data['Target']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 훈련
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # 모델 평가
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 변수 중요도 시각화
    feature_importances = rf_model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, feature_importances)
    plt.title('Feature Importances')
    plt.show()
