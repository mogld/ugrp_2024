import shap

def interpret_shap(model, X_test, target_columns):
    """
    SHAP을 사용한 다중 타겟 모델 해석
    """
    for i, target in enumerate(target_columns):
        print(f"[SHAP 해석: {target}]")
        explainer = shap.TreeExplainer(model.estimators_[i])
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, title=f"SHAP Summary: {target}")
