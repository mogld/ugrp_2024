from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE

def validate_data(X, y):
    """
    데이터 검증 함수: NaN 값 확인 및 오류 처리.
    """
    if X.isnull().values.any():
        raise ValueError("Input X contains NaN.")
    if y.isnull().values.any():
        raise ValueError("Input y contains NaN.")
    print("데이터 검증 완료: NaN 값 없음")


def apply_smote(X_train, y_train, target_columns):
    """
    타겟별로 SMOTE를 개별 적용 후 병합.
    """
    from imblearn.over_sampling import SMOTE
    import pandas as pd

    X_resampled_list = []  # SMOTE로 샘플링된 X 저장
    y_resampled_list = []  # SMOTE로 샘플링된 y 저장

    max_samples = 0  # 최종적으로 사용할 최대 샘플 크기

    for col in target_columns:
        print(f"SMOTE 적용 중: {col}")
        y_target = y_train[col]
        minority_class_count = y_target.value_counts().min()  # 소수 클래스 샘플 수 확인

        if minority_class_count <= 1:
            print(f"SMOTE를 건너뜀: {col} (소수 클래스 샘플 부족: {minority_class_count}개)")
            X_resampled_list.append(X_train)
            y_resampled_list.append(y_target)
        else:
            # k_neighbors 동적 설정
            k_neighbors = min(5, minority_class_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

            # SMOTE 적용
            X_temp, y_temp = smote.fit_resample(X_train, y_target)

            # NaN 값 처리
            if pd.isnull(y_temp).any():
                print(f"경고: {col}에서 SMOTE 결과에 NaN 값 발견, 제거 중...")
                y_temp = pd.Series(y_temp).fillna(0).astype(int)

            # 샘플 크기 업데이트
            max_samples = max(max_samples, len(X_temp))

            # SMOTE 결과 추가
            X_resampled_list.append(X_temp)
            y_resampled_list.append(pd.Series(y_temp, name=col))

    # 모든 타겟 병합 (최대 샘플 크기로 맞춤)
    X_resampled = pd.concat(X_resampled_list, axis=0).iloc[:max_samples].reset_index(drop=True)
    y_resampled = pd.concat(y_resampled_list, axis=1).iloc[:max_samples].reset_index(drop=True)

    # NaN 값 최종 확인 및 처리
    if y_resampled.isnull().values.any():
        print("경고: 병합된 y_resampled에 NaN 값 발견, 제거 중...")
        y_resampled = y_resampled.fillna(0).astype(int)

    print(f"SMOTE 적용 완료: X_train 크기 = {X_resampled.shape}, y_train 크기 = {y_resampled.shape}")
    return X_resampled, y_resampled


def train_disease_risk_model(data, selected_features, target_columns):
    """
    다중 타겟 질환 위험도 예측 모델 학습.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    # 입력 변수(X)와 타겟 변수(y) 분리
    X = data[selected_features]
    y = data[target_columns]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 검증: NaN 값 확인
    validate_data(X_train, y_train)

    # SMOTE 적용
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, target_columns)

    # 데이터 검증: SMOTE 적용 후 NaN 값 확인 및 샘플 크기 일치 확인
    validate_data(X_train_resampled, y_train_resampled)

    if len(X_train_resampled) != len(y_train_resampled):
        raise ValueError("SMOTE 적용 후 X_train_resampled과 y_train_resampled의 크기가 일치하지 않습니다.")

    # 모델 학습
    print("모델 학습 시작...")
    rf_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    rf_model.fit(X_train_resampled, y_train_resampled)

    # 예측 및 평가
    y_pred = rf_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_columns, zero_division=0))

    # AUC-ROC 계산
    try:
        y_pred_proba = rf_model.predict_proba(X_test)
        auc_scores = []
        for i, col in enumerate(target_columns):
            auc = roc_auc_score(y_test[col], y_pred_proba[i][:, 1])
            auc_scores.append(auc)
            print(f"AUC-ROC Score for {col}: {auc:.3f}")
    except Exception as e:
        print(f"AUC-ROC 계산 중 오류 발생: {e}")

    # Precision-Recall Curve 시각화 (선택 사항)
    try:
        for i, col in enumerate(target_columns):
            precision, recall, _ = precision_recall_curve(y_test[col], y_pred_proba[i][:, 1])
            plt.plot(recall, precision, label=f'{col} PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Precision-Recall Curve 시각화 중 오류 발생: {e}")

    return rf_model




