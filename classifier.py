# -*- coding:utf-8 -*-
import h5py
import numpy as np
import pickle
import os
import fire
import tensorflow as tf
from tensorflow.python.client import device_lib
import subprocess
import time

from network import NN
from misc import get_logger, Option
opt = Option('./config.json')

def get_num_gpus():
    n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    return n

def set_num_gpus(use_n_gpus):
    total_n_gpus = get_num_gpus()
    if use_n_gpus > total_n_gpus:
         use_n_gpus = total_n_gpus
    if use_n_gpus < 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(use_n_gpus)])
    if use_n_gpus > 0:
        print("Use {} GPU(s)".format(use_n_gpus))
    else:
        print("Use CPU")
            
    return use_n_gpus


class Classifier:
    
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.h5 = {
                  'train':h5py.File(os.path.join(opt.data_path, 'train.h5'), 'r'),
                  'dev':h5py.File(os.path.join(opt.data_path, 'dev.h5'), 'r'),
                  'test':h5py.File(os.path.join(opt.data_path, 'test.h5'), 'r')}
        self.char_dict = pickle.load(open(os.path.join(opt.data_path, 'char_dict.pkl'), 'rb'), encoding='utf-8')

    def set_dataset(self, use_ratio, train_val_ratio):
        """
        use_ratio: 전체 train 데이터(800만개) 중 얼마나 사용할 지
        train_val_ratio: 사용하는 데이터 중 train:val 비율
        """
        self.use_ratio = use_ratio
        self.train_val_ratio = train_val_ratio 
        total_data_len = self.h5['train']['pid'].shape[0]
        self.train_data_len = int(int(total_data_len * use_ratio) * train_val_ratio)
        self.val_data_len = int(total_data_len * use_ratio) - self.train_data_len
        if self.train_data_len%2 == 1:
            self.train_data_len -= 1
        if self.val_data_len%2 == 1:
            self.val_data_len -= 1
        self.logger.info("전체 train 데이터: %d 중" % total_data_len)
        self.logger.info("%.0f%%(use_ratio)에 해당하는 %d 사용" % (use_ratio*100,
                                                          self.train_data_len+self.val_data_len))
        self.logger.info("train:val = %.0f%%:%.0f%% = %d:%d" % (train_val_ratio*100, 
                                                                  (1-train_val_ratio)*100,
                                                                  self.train_data_len,
                                                                  self.val_data_len))
        
    def get_sample_generator(self, div, batch_size, reverse=False):
        if div == 'train':
            ds = self.h5['train']
            left = 0
            limit = self.train_data_len
        elif div == 'val':
            ds = self.h5['train']
            left = self.train_data_len
            limit = self.train_data_len + self.val_data_len
        else:
            ds = self.h5[div]
            left = 0
            limit = ds['pid'].shape[0]

        if div == 'val' or (div == 'train' and reverse==False):
            while True:
                right = min(left+batch_size, limit)
                X = {col: ds[col][left:right] for col in ['img_feat', 'product_jaso',
                                                         'product_char']}
                Y = {col: ds[col][left:right] for col in ['bcateid', 'mcateid', 
                                                          'scateid', 'dcateid']}
                yield X, Y
                left = right
                if left == limit:
                    if div == 'val':
                        left = self.train_data_len
                    else:
                        left = 0
        elif div == 'train' and reverse==True:
            right, limit = limit, left
            while True:
                left = max(right-batch_size, limit)
                X = {col: ds[col][left:right] for col in ['img_feat', 'product_jaso',
                                                         'product_char']}
                Y = {col: ds[col][left:right] for col in ['bcateid', 'mcateid', 
                                                          'scateid', 'dcateid']}
                yield X, Y
                right = left
                if right == limit:
                    right = self.train_data_len
                
        else:  # div in []'dev', 'test']
            while True:
                right = min(left+batch_size, limit)
                X = {col: ds[col][left:right] for col in ['img_feat', 'product_jaso',
                                                         'product_char']}
                yield X
                left = right
                if left == limit:
                    return

    def train(self, use_n_gpus, resume=-1, model_name='nn', reverse=False):
        """
        resume == 0: not load
        resume == -1: load from latest saved model (not from last epoch model)
        resume > 0: load {resume}th epoch model
        """
        train_gen = self.get_sample_generator('train', opt.batch_size, reverse=reverse)
        train_steps = int(np.ceil(self.train_data_len / opt.batch_size))
        val_gen = self.get_sample_generator('val', opt.batch_size)
        val_steps = int(np.ceil(self.val_data_len / opt.batch_size))
        use_n_gpus = set_num_gpus(use_n_gpus)
        
        nn = NN(model_name, len(self.char_dict))
        with tf.Session() as sess:
            nn.build_model(sess, use_n_gpus)
            sess.run(tf.global_variables_initializer())
            nn.fit(train_gen, train_steps, val_gen, val_steps, opt.num_epochs, resume)
            
    def predict(self, div, epoch=-1, model_name='nn'):
        """
        epoch == -1: load from latest saved model
        """
        data_gen = self.get_sample_generator(div, opt.infer_batch_size)
        total_steps = int(np.ceil(self.h5[div]['pid'].shape[0] / opt.infer_batch_size))
        
        use_n_gpus = set_num_gpus(1)
        nn = NN(model_name, len(self.char_dict))
        with tf.Session() as sess:
            # 주의: predict를 multi gpu에서 실행 시 batch split 시 문제 발생
            nn.build_model(sess, use_n_gpus=use_n_gpus)
            sess.run(tf.global_variables_initializer())
            nn.load(epoch)
            y_preds = nn.predict(data_gen, total_steps)         
        self.write_prediction_result(div, y_preds)

    def ensemble(self, div, *model_epochs):
        """
        model_epoch = ('model_name1', epoch_num, 'model_name2', epoch_num, ...)
        epoch_num이 -1이면 제일 마지막 epoch 사용
        """
        softmax_file_names = []
        print(model_epochs)
        for i in range(0, len(model_epochs), 2):
            model_name = model_epochs[i]
            epoch = model_epochs[i+1]
            self.logger.info('{}: {} epoch'.format(model_name, epoch))
            data_gen = self.get_sample_generator(div, opt.infer_batch_size)
            total_data_len = self.h5[div]['pid'].shape[0]
            total_steps = int(np.ceil(total_data_len / opt.infer_batch_size))
        
            use_n_gpus = set_num_gpus(1)
            nn = NN(model_name, len(self.char_dict))
            with tf.Session(graph=tf.Graph()) as sess:
                nn.build_model(sess, use_n_gpus=use_n_gpus)
                sess.run(tf.global_variables_initializer())
                epoch = nn.load(epoch)
                y_softmaxs = nn.predict(data_gen, total_steps, softmax=True)
                
                file_name = '%s_%dep_%s_softmax.tmp' % (model_name, epoch, div)
                softmax_file_names.append(file_name)
                with h5py.File(file_name, 'w') as fout:
                    fout.create_dataset('b', shape=(total_data_len, opt.num_bcate+1))
                    fout.create_dataset('m', shape=(total_data_len, opt.num_mcate+1))
                    fout.create_dataset('s', shape=(total_data_len, opt.num_scate+1))
                    fout.create_dataset('d', shape=(total_data_len, opt.num_dcate+1))
                    fout['b'][:] = np.array(y_softmaxs['b'])
                    fout['m'][:] = np.array(y_softmaxs['m'])
                    fout['s'][:] = np.array(y_softmaxs['s'])
                    fout['d'][:] = np.array(y_softmaxs['d'])
                del y_softmaxs
        
        # load softmax.tmp, sum, argmax
        chunk_size = 50000
        steps = int(np.ceil(total_data_len / chunk_size))
        y_preds = {'b':[], 'm':[], 's':[], 'd':[]}
        softmax_files = [h5py.File(name, 'r') for name in softmax_file_names]

        for cate in ['b', 'm', 's', 'd']:
            self.logger.info('%s category processing...' % cate)
            
            for i in range(steps):
                
                # softmax h5파일을 chunk_size만큼 읽어서 메모리에 올린다.
                softmax_per_model = []
                for softmax_file in softmax_files:
                    softmax = softmax_file[cate][i*chunk_size:(i+1)*chunk_size]
                    softmax_per_model.append(softmax)
                
                for j in range(len(softmax_per_model[0])):
                    softmax_sum = np.zeros_like(softmax_per_model[0][0])
                    for softmax in softmax_per_model:
                        softmax_sum += softmax[j]
                    y_preds[cate].append(np.argmax(softmax_sum))

                del softmax_per_model
        
        self.write_prediction_result(div, y_preds)

    def write_prediction_result(self, div, y_preds: dict):
        with open('result_%s.tsv'%div, 'w') as fout:
            for pid, b, m, s, d in zip(self.h5[div]['pid'][:], y_preds['b'], y_preds['m'], y_preds['s'], y_preds['d']):
                fout.write('{pid}\t{b}\t{m}\t{s}\t{d}\n'.format(pid=pid.decode(), b=b, m=m, s=s, d=d))

    
if __name__ == '__main__':
    clsf = Classifier()
    clsf.set_dataset(use_ratio=1.0, train_val_ratio=1.0)
    fire.Fire({'train':clsf.train,
               'predict':clsf.predict,
               'ensemble':clsf.ensemble})
