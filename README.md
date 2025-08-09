# **PCOS 관련 질병 위험 예측 AI 모델링**

다낭성 난소 증후군(PCOS)과 관련된 주요 질병의 위험을 예측하기 위한 AI 모델링 프로젝트입니다. 생애 주기별 데이터를 분석하고 머신러닝 모델을 활용하여 개인화된 위험 예측을 제공합니다.

<br>

<div align="center">
  
  [![Python 3.x](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
  [![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/code/jagatheeswari/pcos-dataset)

</div>

---

### **목차**

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [연구 배경](#-연구-배경)
- [데이터](#-데이터)
- [모델링 및 결과](#-모델링-및-결과)
- [SHAP 분석](#-shap-분석)
- [필수 라이브러리](#-필수-라이브러리)
- [연구 보고서](#-연구-보고서)

---

### **<a id="프로젝트-개요"></a> 프로젝트 개요**

이 프로젝트는 다낭성 난소 증후군(PCOS)과 관련된 주요 질병의 위험을 예측하기 위해 AI 기반 모델을 활용합니다. 생애 주기별 주요 위험 요인을 분석하고, 머신러닝 모델을 사용하여 질병의 위험도를 예측함으로써 개인별 예방 및 관리 방안을 모색합니다.

---

### **<a id="주요-기능"></a> 주요 기능**

-   **생애 주기별 분석:** Early, Mid, Late Reproductive 및 Menopause 단계에서의 질병 위험 분석
-   **AI 모델 활용:** **Random Forest**, **XGBoost**, **LightGBM** 모델 적용
-   **SHAP 분석:** 모델 해석력을 높이기 위한 변수 중요도 시각화
-   **데이터 전처리:** 결측치 처리, 타겟 변수 생성 등

---

### **<a id="연구-배경"></a> 연구 배경**

PCOS는 가임기 여성에게 흔히 발생하는 내분비 장애로, **비만**, **제2형 당뇨**, **심혈관계 위험**, **비타민 D 결핍** 등과 밀접한 관련이 있습니다. 이 연구는 AI 모델을 활용해 생애 주기별로 PCOS 관련 질환의 위험도를 예측하고, 개인화된 예방 및 관리 방안을 제공하는 것을 목표로 합니다.

---

### **<a id="데이터"></a> 데이터**

-   **출처:** [Kaggle PCOS Dataset](https://www.kaggle.com/code/jagatheeswari/pcos-dataset)
-   **사용 파일:** `PCOS_infertility.csv`, `PCOS_data_without_infertility.xlsx`
-   **데이터 크기:** 1,082개의 샘플, 51개의 변수
-   **주요 변수:** `BMI` (체질량지수), `AMH`, `FSH`, `LH`, `RBS` 등

---

### **<a id="모델링-및-결과"></a> 모델링 및 결과**

세 가지 AI 모델을 적용하고, **Precision**, **Recall**, **F1-Score**, **AUC-ROC**와 같은 평가 지표를 사용하여 성능을 검증했습니다.

| 모델            | 비만 AUC | 당뇨 AUC | 심혈관 AUC | 비타민 D 결핍 AUC |
| :-------------- | :------- | :------- | :--------- | :---------------- |
| Random Forest   | 1.0      | 0.80     | 0.76       | 0.81              |
| XGBoost         | 1.0      | 0.63     | 0.72       | 0.78              |
| LightGBM        | 1.0      | 0.66     | 0.74       | 0.78              |

---

### **<a id="shap-분석"></a> SHAP 분석**

SHAP(SHapley Additive exPlanations) 분석을 통해 각 모델의 예측에 영향을 미친 주요 변수들을 시각화했습니다.

-   **비만:** **BMI**가 가장 중요한 변수로 나타났습니다.
-   **당뇨:** **LH**와 **BMI**가 주요 예측 변수였습니다.
-   **심혈관계 위험:** **FSH**, **BMI**, **LH**가 중요한 영향을 미쳤습니다.
-   **비타민 D 결핍:** **FSH**와 **LH**가 핵심 변수였습니다.

---

### **<a id="필수-라이브러리"></a> 필수 라이브러리**

이 프로젝트를 실행하는 데 필요한 라이브러리 목록입니다. pip를 통해 설치할 수 있습니다.

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib
```
---
### **<a id="연구-보고서"></a> 연구 보고서**
[PCOS UGRP 결과보고서](./PCOS-ugrp결과보고서.pdf)
