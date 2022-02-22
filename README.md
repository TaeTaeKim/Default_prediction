# 채무 불이행 예측

## 분석개요
    1. 데이터 Upsampling을 위한 SMOTE라이브러리 사용 Target 비율을 0,5:0.5로 맞춤
    2. 통계 기반 XGboost, LightGBM, SVM, Bagging, AdaBoost 등 사용
    3. 딥러닝 기반 Python로 이진 분류 MLP 구현
    4. Hyper Parameter 튜닝을 위한 GridSearch, RandomizedSearch를 사용
    5. F1-Score 향상을 위한 Threshold 조정
    
## 1. 데이터 전처리
Data Split : 0.8:0.2로 나눔
```python 
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1001)
```

SMOTE를 활용한 Train Dataset Upsampling
```python
smote = SMOTE(sampling_strategy='minority',random_state=1001)
```

범주형 변수로 사용될 수 있는 값을 제외한 변수들 정규화
```python
#표준화가 필요한 컬럼들
to_norm = ['int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
       'pub_rec', 'revol_bal', 'total_acc', 'collections_12_mths_ex_med',
       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal',
       'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens', 'funded_amnt', 'funded_amnt_inv',
       'total_rec_late_fee', 'term1', 'open_acc', 'installment', 'revol_util',
       'out_prncp', 'out_prncp_inv', 'total_rec_int', 'fico_range_low',
       'fico_range_high']
```

Processing 결과
* 원본 데이터 target의 비율 : 0.32569
* oversampled_train 데이터 target의 비율 : 0.5
* validation 데이터 target의 비율 : 0.3258
## Xgboost
모델과 Parameter grid정의
```python
xgb = xgboost.XGBClassifier(use_label_encoder=False,objective='binary:logistic',tree_method='gpu_hist',
                           predictor='gpu_predictor',eval_metric='mlogloss')
param_grid = {
    'learning_rate':[0.01,0.3,0.5,0.8],
    'gamma':[0,1,2],
    'max_depth': [3,5,7],
    'min_child_weight':[0,1,2],
    'subsample': [0.5,1],
    'colsample_bytree':[0.5,1],
    "n_estimators":[100,300]
             }
```
* Optimized Model Performance

![XGBoost_ACC](https://user-images.githubusercontent.com/70123707/155203136-e07451ec-3851-4744-be47-3d76443589d5.png)

    - 정확도: 0.7450
    - 정밀도: 0.6410
    - 재현율: 0.4943
    - AUC: 0.6803
    - F1: 0.5582
    - {'subsample': 1, 'n_estimators': 300, 'min_child_weight': 2, 'max_depth': 5, 'learning_rate': 0.3, 'gamma': 0, 'colsample_bytree': 1}

* Threshold 튜닝을 이용한 F1 Score최적화

![XGBoost_Threshold](https://user-images.githubusercontent.com/70123707/155203476-05a1b2e0-74fe-4507-ab05-cef8c9e6fb85.png)

    - Threshold : 0.29 | F1_score : 0.63
## SVM
* Hyper parameter C grid : [0.1,1,3,5,7,9,10,11,12,13]
* Graph

![SVM_Performance](https://user-images.githubusercontent.com/70123707/155204583-70886179-3077-4129-8b2f-ec62c3a3674d.png)
- C = 12 Model 생성 후 Probability와 Threshold를 이용한 F1-score 최적화
![image](https://user-images.githubusercontent.com/70123707/155204815-0992a984-4b9f-4af8-9117-ff782deb20b0.png)

    - Threshold : 0.33 | F1_score : 0.60
## 3. LightGBM
모델과 Parameter Grid
```python
lgbm = LGBMClassifier(objective='binary')
lgbm_params = {
    'learning_rate':[0.01,0.05,0.1,0.3,0.5],
    'n_estimators':[100,200,300,400],
    'max_depth':[3,5,7],
    'min_child_samples':[20,25,30],
    'subsample':[0.5,0.6,0.7],
    'colsample_bytree':[0.5,0.5,1],
    'metric':['binary_logloss','cross_entropy']
}
```
* Optimized Model Performance

![image](https://user-images.githubusercontent.com/70123707/155205467-1047b157-c969-4a8e-81da-b768e5e97b93.png)

    - 정확도: 0.7531
    - 정밀도: 0.6616
    - 재현율: 0.4957
    - AUC: 0.6866
    - F1: 0.5668
    - {'subsample': 0.5, 'n_estimators': 400, 'min_child_samples': 25, 'metric': 'binary_logloss', 'max_depth': 7, 'learning_rate': 0.05, 'colsample_bytree': 1}

* Threshold 튜닝을 이용한 F1 Score최적화

![image](https://user-images.githubusercontent.com/70123707/155205640-88add5b6-33ce-4076-a7b2-bb348ead2495.png)

    - Threshold At 0.28 | F1_score : 0.64
* Tuning 후 성능

![image](https://user-images.githubusercontent.com/70123707/155206248-2c84c23a-72db-4a19-bebd-a40f94afc45b.png)

    - 정확도: 0.7100
    - 정밀도: 0.5379
    - 재현율: 0.7816
    - AUC: 0.7285
    - F1: 0.6372
    
## Bagging
### 1. Bagging을 위한 Decision Tree 생성과 최적화
* Decision Tree parameter grid
```python
des_tree = DecisionTreeClassifier()
num_features = len(overed_X.columns)
des_tree_param = {
    'criterion':['gini','entropy'],
    'max_depth':[np.round(num_features*0.5),np.round(num_features*0.7),np.round(num_features*0.9)],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,5,10]
}
```
* Optimized Deicision Tree Performance

![image](https://user-images.githubusercontent.com/70123707/155207117-56ea507f-6848-4f48-bddd-3082cecd94a2.png)

    - 정확도 : 0.6905
    - 정밀도 : 0.5273
    - 재현율 : 0.4830
    - AUC    : 0.6369
    - F1     : 0.5042
    - {'criterion': 'gini', 'max_depth': 38.0, 'min_samples_leaf': 10, 'min_samples_split': 2}
    
* Threshold Tuning

![image](https://user-images.githubusercontent.com/70123707/155207378-ab0b7c27-0074-4df3-8212-61ccbc117c6c.png)

    - Threshold At 0.16 | F1_score : 0.56
### 2. Bagging을 활용한 예측, Parameter 수정
* bagging Model parameter grid
```python
bagging = BaggingClassifier(base_estimator=des_tree,n_jobs=-1)
bag_param = {
    'n_estimators':[10,50,100,200,300],
    'max_samples':[0.3,0.5,0.7,0.9],
    'max_features':[0.3,0.5,0.7,0.9]
}
```
* Optimized Bagging Model Performance

![image](https://user-images.githubusercontent.com/70123707/155207598-788ee301-3f40-45d0-869f-fb7889373717.png)

    - 정확도 : 0.7464
    - 정밀도 : 0.7148
    - 재현율 : 0.3688
    - AUC    : 0.6488
    - F1     : 0.4865
    - {'max_features': 0.7, 'max_samples': 0.9, 'n_estimators': 300}
    
* Threshold tuning

![image](https://user-images.githubusercontent.com/70123707/155207771-6b697f26-8695-4f5c-8709-a068af8eab82.png)

    - Threshold At 0.32 | F1_score : 0.64


## AdaBoost
* Model & Parameter Grid
```python
adaboost = AdaBoostClassifier(base_estimator=des_tree)
ada_param = {
    'n_estimators':[50,100,200],
    'learning_rate':[0.1,0.5,1]
}
```

* Optimized Model Performance

![image](https://user-images.githubusercontent.com/70123707/155208277-da8dae56-5392-43f6-a943-2703a996bd3a.png)

    - 정확도: 0.7356
    - 정밀도: 0.6385
    - 재현율: 0.4340
    - AUC: 0.6576
    - F1: 0.5168
    - {'learning_rate': 0.1, 'n_estimators': 200}
    
![image](https://user-images.githubusercontent.com/70123707/155208463-06086a81-944d-48c5-b513-04c689fbb812.png)

    - Threshold At 0.11 | F1_score : 0.61
    
## DNN MLP
* MLP모델 정의
    - 75(features) - 1024 - 1024 - 1024 - 512 - 256 - 256 - 128 - 64 - 1 의 node를 가짐
    - 매 layer 중간에 BatchNorm & Dropout(0.5)를 적용
    - 마지막은 Sigmoid함수 적용
    - Epoch = 30 , BatchSize = 128, Loss Func: BCELoss, Optimizer : Adam(lr=0.01)
    - 매 Epoch마다 Validation Performance를 저장하고 가장 높은 F1성능을 저장.
 * Model Performace
 
 ![image](https://user-images.githubusercontent.com/70123707/155209333-c2784309-e44f-43ae-9da8-80c52c644621.png)
 
    - Best F1 score : 0.62
    
* Threshold 조정
    - [[9258 4226]
    - [1453 5063]] 0.6406833280607402
    - [[10595  2889]
    - [ 2212  4304]] 0.6279086731344372
    - [[11763  1721]
    - [ 3155  3361]] 0.5795826866701155
    - [[12686   798]
    -  [ 4300  2216]] 0.46505771248688355
    - [[13163   321]
    - [ 5126  1390]] 0.3379117539807949

 ## 최종 Test Data에 대한 Predciton
 * model score를 Weight한 Soft voting Prediction **최종성능 : 0.6983**
 * hard voting Prediction                        **최종성능 : 0.7041**
 * 각각의 모델을 이용한 Prediction               **최종성능 : 0.7167**
 
 
 
