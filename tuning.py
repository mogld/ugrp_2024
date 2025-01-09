from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_hyperparameters(X_train, y_train):
    """
    랜덤 포레스트 모델의 하이퍼파라미터를 최적화합니다.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
