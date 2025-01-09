import pandas as pd

def save_results(results, file_name):
    """
    모델 결과를 CSV 파일로 저장합니다.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")