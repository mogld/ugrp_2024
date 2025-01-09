import pandas as pd
from preprocessing import preprocess_data
from eda import perform_eda, analyze_by_life_stage
from modeling import train_disease_risk_model
from visualization import visualize_risk_by_life_stage
from interpretation import interpret_shap

# 1. 데이터 로드
infertility_data = pd.read_csv('data/PCOS_infertility.csv')
non_infertility_data = pd.read_excel('data/PCOS_data_without_infertility.xlsx', sheet_name=1)

# 2. 타겟 변수 정의
infertility_data['Target'] = infertility_data['PCOS (Y/N)']
non_infertility_data['Target'] = non_infertility_data['PCOS (Y/N)']

# 3. 데이터 병합 및 전처리
data = pd.concat([infertility_data, non_infertility_data], axis=0, ignore_index=True)
data = preprocess_data(data)

# 4. 탐색적 데이터 분석 (EDA)
print("\n[탐색적 데이터 분석]")
perform_eda(data)
analyze_by_life_stage(data, ['BMI', 'FSH(mIU/mL)', 'AMH(ng/mL)'])

# 주요 질환 타겟 변수 추가
print("\n[질환 타겟 변수 추가]")
data['Obesity'] = (data['BMI'] > 30).astype(int)
data['Type2_Diabetes'] = (data['RBS(mg/dl)'] > 126).astype(int)
data['Cardiovascular_Risk'] = ((data['BP _Systolic (mmHg)'] > 130) | (data['BP _Diastolic (mmHg)'] > 85)).astype(int)
data['Vitamin_D_Deficiency'] = (data['Vit D3 (ng/mL)'] < 20).astype(int)

# 새로운 타겟 변수
target_columns = ['Obesity', 'Type2_Diabetes', 'Cardiovascular_Risk', 'Vitamin_D_Deficiency']

# 질환 위험도 예측 모델 학습
print("\n[질환 위험도 예측 모델 학습]")
selected_features = [
    'Age (yrs)', 'BMI', 'AMH(ng/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
    'TSH (mIU/L)', 'Life_Stage_Early_Reproductive', 'Life_Stage_Mid_Reproductive',
    'Life_Stage_Late_Reproductive', 'Life_Stage_Menopause'
]
rf_model = train_disease_risk_model(data, selected_features, target_columns)

# 7. SHAP을 사용한 모델 해석
print("\n[SHAP 해석]")
interpret_shap(rf_model, data[selected_features], target_columns)

# 8. 생애 주기별 질환 위험도 시각화
print("\n[생애 주기별 질환 위험도 시각화]")
visualize_risk_by_life_stage(data, rf_model, selected_features, target_columns)

# 모델의 예측 결과 추가
print("\n[모델 예측 결과 저장]")
data['Predicted_Obesity'] = rf_model.predict(data[selected_features])[:, 0]
data['Predicted_Type2_Diabetes'] = rf_model.predict(data[selected_features])[:, 1]
data['Predicted_Cardiovascular_Risk'] = rf_model.predict(data[selected_features])[:, 2]
data['Predicted_Vitamin_D_Deficiency'] = rf_model.predict(data[selected_features])[:, 3]

# 전체 데이터를 CSV 파일로 저장
data.to_csv('data/PCOS_analysis_results.csv', index=False, encoding='utf-8')
print("결과가 'data/PCOS_analysis_results.csv' 파일에 저장되었습니다.")


