import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def define_life_stage(age):
    """
    생애 주기 정의 함수
    """
    if age < 19:
        return 'Adolescence'
    elif age < 26:
        return 'Early_Reproductive'
    elif age < 36:
        return 'Mid_Reproductive'
    elif age < 46:
        return 'Late_Reproductive'
    else:
        return 'Menopause'


def preprocess_data(data):
    """
    데이터 전처리 함수: 결측치 처리, 범주형 인코딩, 생애 주기 정의 등
    """
    data.columns = data.columns.str.strip()  # 열 이름 정리

    # 숫자형 변환 및 결측치 처리
    numeric_cols = ['Age (yrs)', 'AMH(ng/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'RBS(mg/dl)', 'TSH (mIU/L)']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 생애 주기 열 추가
    data['Life_Stage'] = data['Age (yrs)'].apply(define_life_stage)

    # 숫자형 열 결측치 처리
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    data[numeric_data.columns] = numeric_imputer.fit_transform(numeric_data)

    # 범주형 열 선택 및 문자열로 변환
    categorical_cols = data.select_dtypes(include=['object']).columns.difference(['Life_Stage'])
    for col in categorical_cols:
        data[col] = data[col].astype(str)

    # 범주형 열 결측치 처리
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    # 범주형 데이터 원-핫 인코딩
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_life_stage = pd.DataFrame(
        encoder.fit_transform(data[['Life_Stage']]),
        columns=encoder.get_feature_names_out(['Life_Stage'])
    )

    # 원본 데이터와 병합
    data = pd.concat([data, encoded_life_stage], axis=1)

    # 결측치 제거 확인
    if data.isnull().sum().sum() > 0:
        print("경고: 일부 결측치가 여전히 남아 있습니다. 확인 필요!")
    else:
        print("모든 결측치 처리 완료!")

    print("전처리 완료: 데이터 셋 크기 ->", data.shape)
    return data

