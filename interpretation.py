import shap
import matplotlib.pyplot as plt

def interpret_model(model, X_test):
    """
    SHAP을 사용해 모델의 결과를 해석합니다.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary Plot
    shap.summary_plot(shap_values[1], X_test)

    # Feature Importance
    shap.summary_plot(shap_values[1], X_test, plot_type='bar')
