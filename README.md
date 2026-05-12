# Credit Score Classification with TabTransformer

## 프로젝트 개요
고객의 금융 정보를 바탕으로 신용점수 등급을 예측하는 딥러닝 기반 다중분류 프로젝트입니다.  
예측 대상은 `Credit_Score`이며, 클래스는 `Good`, `Standard`, `Poor`입니다.

기본 MLP 모델을 먼저 구축한 뒤, 범주형 변수와 수치형 변수를 분리 처리하는 `TabTransformer`를 적용하여 성능을 개선했습니다.

## 데이터 출처
Kaggle Credit Score Classification  
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## EDA

### 결측치 확인
전체 결측치 개수를 확인한 결과 0개로 나타났습니다.  
따라서 별도의 결측치 대체나 삭제는 수행하지 않았습니다.

<img width="414" height="462" alt="null_check" src="https://github.com/user-attachments/assets/e8706813-eba5-4695-9a5d-2056951a0655" />

### Credit_Score 클래스 분포
`Credit_Score`는 `Standard`, `Poor`, `Good` 세 클래스로 구성되어 있습니다.  
분포를 확인한 결과 `Standard` 클래스가 가장 많고, `Good` 클래스가 가장 적었습니다.  
따라서 모델이 다수 클래스에 편향될 가능성이 있어 최종 모델에서는 class weight를 적용했습니다.

### 주요 수치형 변수 분포
소득, 이자율, 미지급 잔액, 월별 잔고 등 주요 수치형 변수들의 분포를 확인했습니다.  
변수마다 값의 범위가 다르기 때문에 모델 학습 전 수치형 변수에는 `StandardScaler`를 적용했습니다.

### Credit_Score별 주요 변수 차이
`Outstanding_Debt`와 `Interest_Rate`는 신용점수 등급별로 뚜렷한 차이를 보였습니다.  
`Good` 등급은 상대적으로 낮은 부채와 낮은 이자율에 분포했고, `Poor` 등급은 높은 부채와 높은 이자율에 분포하는 경향이 있었습니다.

<img width="704" height="470" alt="outstanding_debt" src="https://github.com/user-attachments/assets/ef4af4c0-d2db-437d-a8b9-520245c2de87" />

<img width="686" height="470" alt="interest_rate" src="https://github.com/user-attachments/assets/71093827-968b-4405-abf3-c3e6484268d4" />

### 상관관계 분석
상관관계 히트맵을 통해 수치형 변수들 간의 관계를 확인했습니다.

<img width="1383" height="1190" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/5884f4a5-0e37-46ee-a92a-a65b46635d50" />

`Annual_Income`과 `Monthly_Inhand_Salary`는 강한 양의 상관관계를 보였습니다.  
또한 `Interest_Rate`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Outstanding_Debt`는 서로 양의 상관관계를 보여 연체, 이자율, 미지급 잔액 관련 변수들이 함께 증가하는 경향을 확인했습니다.  
반면 `Credit_History_Age`는 위험 관련 변수들과 음의 상관관계를 보여 신용 이력이 길수록 연체 및 미상환 부채 관련 위험이 낮아지는 경향을 확인했습니다.

## 전처리

- `ID`, `Customer_ID`, `Name`, `SSN` 제거
- 범주형 변수와 수치형 변수 분리
- 수치형 변수는 train 데이터에만 `StandardScaler`를 fit한 뒤 validation 데이터에는 transform만 적용
- 데이터 누수 방지를 위해 scaler는 train split 이후 적용
- `Credit_Score`는 `LabelEncoder`로 인코딩
- `Type_of_Loan`은 고유 문자열 조합이 6,261개로 많아 원본 컬럼은 제거하고 파생변수로 변환

## 파생변수 생성

신용 위험을 더 직접적으로 표현하기 위해 다음과 같은 파생변수를 추가했습니다.

### 소득 및 상환 부담 관련 변수
- `Debt_to_Annual_Income`
- `Debt_to_Monthly_Salary`
- `EMI_to_Monthly_Salary`
- `Invest_to_Monthly_Salary`
- `Outflow_to_Monthly_Salary`
- `Disposable_Ratio`

### 연체 및 신용 위험 관련 변수
- `Delayed_Payment_Ratio`
- `Delay_per_Delayed_Payment`
- `Inquiry_per_Card`
- `Debt_per_Credit_History`
- `Credit_History_per_Age`
- `Utilization_Debt_Interaction`

### Type_of_Loan 기반 파생변수
`Type_of_Loan`은 고유 문자열 조합이 6,261개로 많아 원본 문자열을 그대로 사용하면 과적합 위험이 있다고 판단했습니다.  
따라서 원본 컬럼은 제거하고 다음 파생변수만 사용했습니다.

- `Loan_Type_Count`
- `Has_Credit_Builder_Loan`
- `Has_Personal_Loan`
- `Has_Mortgage_Loan`
- `Has_Auto_Loan`
- `Has_Student_Loan`
- `Has_Home_Equity_Loan`
- `Has_Debt_Consolidation_Loan`
- `Has_Payday_Loan`

## 모델링

### Baseline: MLP
기본 모델은 64-32 구조의 MLP로 구성했습니다.  
범주형 변수는 `LabelEncoder`로 숫자화한 뒤 수치형 변수와 함께 입력했습니다.

```text
Input
→ Linear(input_size, 64)
→ ReLU
→ Linear(64, 32)
→ ReLU
→ Linear(32, 3)
```

기본 MLP는 구현이 간단하지만, Label Encoding된 범주형 변수를 수치형 변수처럼 처리하기 때문에 범주 간 순서가 있는 것처럼 해석될 수 있다는 한계가 있습니다.

### Final Model: TabTransformer
최종 모델은 `TabTransformer`를 사용했습니다.  
TabTransformer는 범주형 변수를 embedding으로 변환한 뒤 self-attention을 통해 범주형 변수 간 관계를 학습할 수 있습니다.  
수치형 변수는 파생변수 추가 후 정규화하여 별도 입력으로 사용했습니다.

주요 설정은 다음과 같습니다.

```text
dim = 32
depth = 4
heads = 4
attn_dropout = 0.2
ff_dropout = 0.2
batch_size = 256
learning_rate = 0.001
weight_decay = 0.0001
early_stopping patience = 7
```

클래스 불균형을 보정하기 위해 `CrossEntropyLoss`에 class weight를 적용했습니다.  
특히 `Good` 클래스의 표본 수가 상대적으로 적어, 해당 클래스가 학습 과정에서 덜 무시되도록 loss 비중을 조정했습니다.

## 개선 사항 및 결과

기존 MLP는 범주형 변수를 `LabelEncoder`로 숫자화한 뒤 수치형 변수와 함께 스케일링하여 입력했습니다.  
이는 구현이 간단하지만 범주 간 순서가 있는 것처럼 해석될 수 있고, 복잡한 범주형 정보를 충분히 반영하기 어렵습니다.

개선 모델에서는 범주형 변수와 수치형 변수를 분리 처리하는 `TabTransformer`를 사용했습니다.  
또한 부채 대비 소득, 월급 대비 상환 부담, 연체 비율, 신용 거래 기간 관련 파생변수를 추가했습니다.  
`Type_of_Loan`은 고유 문자열 조합이 6,261개로 많아 원본 대신 대출 개수와 주요 대출 유형 여부 변수로 변환했습니다.  
추가로 class weight와 early stopping을 적용하여 클래스 불균형과 과적합을 완화하고자 했습니다.

### 모델 성능 비교

| 모델 | 주요 설정 | Validation Accuracy |
| --- | --- | ---: |
| 기본 MLP | 64-32 hidden layer, batch size 32, epoch 20 | 71.70% |
| 개선 MLP | 256-128-64 hidden layer, BatchNorm, Dropout, batch size 128 | 74.65% |
| 개선 MLP + 추가 학습 | learning rate 0.0005, 추가 5 epoch | 75.42% |
| TabTransformer | 파생변수, class weight, early stopping 적용 | 80.56% |

최종적으로 TabTransformer 기반 개선 모델은 `80.56%`의 최고 Validation Accuracy를 기록했습니다.  
기본 MLP 대비 `+8.86%p`, 개선 MLP + 추가 학습 대비 `+5.14%p` 향상되었습니다.

### Classification Report

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Good | 0.73 | 0.82 | 0.77 |
| Poor | 0.79 | 0.83 | 0.81 |
| Standard | 0.85 | 0.79 | 0.82 |
| Weighted Avg | 0.81 | 0.81 | 0.81 |

Class weight 적용 후 `Good` 클래스의 recall이 개선되어, 소수 클래스에 대한 예측 성능도 함께 보완되었습니다.

## 사용 기술 스택

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- TabTransformer
- Google Colab
