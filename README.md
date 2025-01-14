# PCOS 관련 질병 위험 예측 AI 모델링

## 📖 프로젝트 개요
이 프로젝트는 다낭성 난소 증후군(PCOS)과 관련된 주요 질병 위험을 예측하기 위해 AI 기반 모델을 활용합니다.  
생애 주기별 분석을 통해 PCOS 관련 질병의 주요 위험 요인을 분석하고, AI 기반 모델을 사용하여 주요 질병의 위험도를 예측합니다. 

---

## 📌 주요 기능
- **생애 주기별 분석**  
  Early, Mid, Late Reproductive 및 Menopause 단계에서 질병 위험 분석
- **AI 모델 활용**  
  Random Forest, XGBoost, LightGBM 적용
- **SHAP 분석**  
  모델 해석력을 높이기 위한 변수 중요도 시각화
- **데이터 전처리**  
  결측치 처리, 타겟 변수 생성 등

---
## 📄 연구 보고서 PDF
[PCOS 연구 최종 보고서](./PCOS_연구_최종_보고서.pdf)
<embed src="./PCOS_연구_최종_보고서.pdf" width="100%" height="600px" />

---

## 📂 목차
1. [연구 배경](#-연구-배경)
2. [데이터](#-데이터)
3. [전처리](#-전처리)
4. [모델링](#-모델링)
5. [결과](#-결과)
6. [필수 라이브러리](#-필수-라이브러리)
7. [향후 연구 방향](#-향후-연구-방향)

---

## 🔍 연구 배경
PCOS는 가임기 여성에게 흔히 발생하는 내분비 장애로, 다음과 같은 질병들과 밀접한 관련이 있습니다:
- 비만
- 제2형 당뇨
- 심혈관계 위험
- 비타민 D 결핍

특히 주변 20대 친구들에게도 생각보다 흔히 발견되는 질환이라는 점에서, 본 연구의 필요성이 더욱 강조되었습니다. 본 연구는 AI모델을 활용하여 생애 주기별 (초기 가임기, 중기 가임기, 후기 가임기, 폐경기)로 PCOS와 관련된 질환의 위험도를 예측하고, 이를 통해 개인화된 예방 및 관리 방안을 제공하고자 합니다. 



---

## 📊 데이터
- **출처:** [Kaggle PCOS Dataset](https://www.kaggle.com/code/jagatheeswari/pcos-dataset)
- **사용된 파일:**
  - `PCOS_infertility.csv`
  - `PCOS_data_without_infertility.xlsx`
- **데이터 크기:** 1,082개의 샘플, 51개의 변수
- **주요 변수:**
  - BMI (체질량지수)
  - AMH (Anti-Müllerian Hormone)
  - FSH (Follicle-Stimulating Hormone)
  - LH (Luteinizing Hormone)
  - RBS (Random Blood Sugar)

---

## 🛠 전처리
- **결측치 처리:** 평균(mean) 및 최빈값(most frequent)으로 대체
- **생애 주기 변수 생성:** 나이를 기준으로 Life Stage 변수 생성
- **타겟 변수 정의:**
  - 비만(Obesity): BMI > 30
  - 당뇨(Type 2 Diabetes): RBS > 126
  - 심혈관계 위험(Cardiovascular Risk): Systolic BP > 130 또는 Diastolic BP > 85
  - 비타민 D 결핍(Vitamin D Deficiency): Vitamin D3 < 20

---

## 🤖 모델링
- **적용 모델:**
  - Random Forest
  - XGBoost
  - LightGBM
- **평가 지표:**
  - Precision, Recall, F1-Score, AUC-ROC
- **SHAP 분석:** 변수 중요도 시각화를 통한 해석력 확보

---

## 📈 결과
### **모델 성능**
| 모델          | 비만 AUC | 당뇨 AUC | 심혈관 AUC | 비타민 D 결핍 AUC |
|---------------|----------|----------|------------|------------------|
| Random Forest | 1.0      | 0.80     | 0.76       | 0.81             |
| XGBoost       | 1.0      | 0.63     | 0.72       | 0.78             |
| LightGBM      | 1.0      | 0.66     | 0.74       | 0.78             |

### **SHAP 분석 결과**
- **비만(Obesity):** BMI가 가장 중요한 변수
- **당뇨(Type 2 Diabetes):** LH와 BMI가 주요 변수
- **심혈관계 위험(Cardiovascular Risk):** FSH, BMI, LH
- **비타민 D 결핍(Vitamin D Deficiency):** FSH, LH

---

## 🚀 필수 라이브러리

- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `matplotlib`


---

## 🔮 향후 연구 방향
- 데이터 증강(SMOTE) 적용으로 소수 클래스 문제 해결
- 임상 데이터로 외부 검증 수행
- 생애 주기별 맞춤형 예방 프로그램 개발

