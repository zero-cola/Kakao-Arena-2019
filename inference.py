import classifier

if __name__ == '__main__':
    clsf = classifier.Classifier()
    clsf.set_dataset(use_ratio=1.0, train_val_ratio=1.0)
    # ./model 디렉토리 안에 JCCNN_0, JCCNN_1, JCCNN_2 이라는 디렉토리가 있어야 함
    # 또한 각 폴더안에 학습된 모델이 존재해야 함. (model_1.ckpt.data, model_1.ckpt.index, ...)
    # -1은 각 폴더안의 모델 중 가장 최근 epoch의 모델을 inference에 사용한다는 것을 의미함
    clsf.ensemble('test', 'JCCNN_0', -1, 'JCCNN_1', -1, 'JCCNN_2', -1)

