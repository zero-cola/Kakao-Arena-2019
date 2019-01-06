import classifier

if __name__ == '__main__':
    clsf = classifier.Classifier()
    clsf.set_dataset(use_ratio=1.0, train_val_ratio=1.0)
    clsf.train(use_n_gpus=1, resume=-1, model_name='JCCNN_0', reverse=False) # 1개의 gpu사용, 이전 모델 있으면 이어서 학습
    
