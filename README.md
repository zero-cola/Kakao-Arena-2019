# Kakao-Arena-2019

쇼핑몰 상품 카테고리 분류 

https://arena.kakao.com/c/1

Team : 제로콜라

Download pre-trained ensemble model link [here](https://drive.google.com/open?id=1oez_YRMno0pw1ps1Wm03n_aPAYsgojJ4)


# 개발 환경
OS: ubuntu
PYTHON: python 3.6.7
        tensorflow-gpu 1.12.0
        
GPU: GeForce GTX 1080ti


# 실행 방법    

### 1. data 준비
1-1. chunk로 나뉘어진 데이터를 적절한 위치에 두고, config.json에 해당 경로를 설정한다.
   
1-2. 아래 명령어를 각각 실행한다. (config.json에 설정된 data_path에 새로운 데이터 파일이 생성됨. 총 90GB)
   
     python data.py make-h5 train
     python data.py make-h5 dev
     python data.py make-h5 test
       
1-3. 아래 명령어를 순서대로 실행한다. (text데이터에 대한 전처리를 진행하며 새로운 데이터 파일이 생성됨. 총 150GB)
   
     python data.py preprocessing train
     python data.py preprocessing dev
     python data.py preprocessing test
        

### 2. 학습

최종 제출은 3개의 모델을 앙상블하여 예측결과를 생성하였다.
아래는 각 모델의 학습을 실행하는 방법이며,
각각 다른 환경에서 병렬적으로 실행한 뒤 inference할 환경에 학습된 모델을 옮겨서 앙상블 할 수 있다.
     
2-1. JCCNN_0 모델 학습
     
     train.py 코드 내부의 model_name 인자를 'JCCNN_0'로 바꿔주고,
     config.json의 dropout_keep의 값을 0.5로 설정한다.
     그 후 python train.py 를 실행한다.
     
2-2. JCCNN_1 모델 학습
     
     train.py 코드 내부의 model_name 인자를 'JCCNN_1'로 바꿔주고,
     config.json의 dropout_keep의 값을 0.6로 설정한다.
     그 후 python train.py 를 실행한다.
     
2-3. JCCNN_2 모델 학습
     
     train.py 코드 내부의 model_name 인자를 'JCCNN_2'로 바꿔주고,
     config.json의 dropout_keep의 값을 0.7로 설정한다.
     그 후 python train.py 를 실행한다.
     
epoch이 끝날 때 마다 모델이 ./model/[model_name]에 저장된다.
학습 도중 중단되어도 위의 명령어를 다시 입력하면 최근 저장된 epoch부터 이어서 학습한다.


### 3. 예측

세 모델이 모두 학습되어 ./model/JCCNN_0, ./model/JCCNN_1, ./model/JCCNN_2에 저장되어 있어야 한다.
     
3-1. python inference.py
     
     test데이터에 대해 inference 진행된다.
     완료되면 현재 디렉터리에 result_test.tsv 파일이 생성된다.
     (중간 생성 파일(.tmp)이 존재하기 때문에 72GB의 저장공간이 필요하다.)
