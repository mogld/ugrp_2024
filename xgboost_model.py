from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    XGBoost 모델을 훈련하고 평가합니다.
    """
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

    print("XGBoost Accuracy:", accuracy)
    print("XGBoost ROC AUC Score:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
