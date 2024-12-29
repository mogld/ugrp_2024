from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def preprocess_data(data):
    """
    데이터 전처리를 수행합니다: 생애 주기 정의, 결측치 처리, 범주형 인코딩 등.
    """
    # 생애 주기 정의
    def define_life_stage(age):
        if age < 19:
            return 'Adolescence'
        elif age < 26:
            return 'Early Reproductive'
        elif age < 36:
            return 'Mid Reproductive'
        elif age < 46:
            return 'Late Reproductive'
        else:
            return 'Menopause'

    if 'Age' in data.columns:
        data['Life Stage'] = data[' Age (yrs)'].apply(define_life_stage)

    # 불필요한 열 제거
    data = data.loc[:, ~data.columns.str.contains('Unnamed')]
    data = data.dropna(axis=1, how='all')
    data['AMH(ng/mL)'] = pd.to_numeric(data['AMH(ng/mL)'], errors='coerce')

    print("Missing Values Before:\n", data.isnull().sum())

    # 결측치 처리 (숫자는 평균, 범주는 최빈값으로 대체)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # 범주형 열을 문자열로 변환
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype(str)  # 문자열로 변환하여 혼합된 데이터 문제 해결

    # 범주형 데이터의 결측치 처리
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    # 범주형 데이터 인코딩
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    encoded_data.columns = encoder.get_feature_names(categorical_cols)

    # 원래 데이터와 병합
    data = pd.concat([data.drop(categorical_cols, axis=1), encoded_data], axis=1)

    print("Preprocessed Data Shape:", data.shape)
    return data

