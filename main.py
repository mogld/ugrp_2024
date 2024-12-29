# 프로젝트 파일 구조를 기반으로 코드를 파일별로 나누어 구성

# main.py
import pandas as pd
from preprocessing import preprocess_data
from eda import perform_eda
from modeling import train_and_evaluate_model

# 데이터 로드
infertility_data = pd.read_csv('data/PCOS_infertility.csv')
non_infertility_data = pd.read_excel('data/PCOS_data_without_infertility.xlsx', sheet_name=1)


# Target 변수 정의
infertility_data['Target'] = infertility_data['PCOS (Y/N)']
non_infertility_data['Target'] = non_infertility_data['PCOS (Y/N)']

# 데이터 전처리
data = pd.concat([infertility_data, non_infertility_data], axis=0, ignore_index=True)
data = preprocess_data(data)


# 입력 변수와 종속 변수 정의
selected_features = [' Age (yrs)', 'BMI', 'AMH(ng/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'RBS(mg/dl)', 'TSH (mIU/L)','PCOS (Y/N)']
X = data[selected_features]
y = data['Target']


# 탐색적 데이터 분석
perform_eda(data)

# 모델 훈련 및 평가
train_and_evaluate_model(data)

