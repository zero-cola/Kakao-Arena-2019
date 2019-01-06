import tensorflow as tf
import numpy as np
import os
import re
import sys
import time
from misc import get_logger, Option
import tqdm
import json
opt = Option('./config.json')


def make_parallel(fn, num_gpus, **kwargs):

    if num_gpus < 2:
        return fn(**kwargs)
    
    # Data-parallel, loss average방식 (기존 배치크기를 num_gpus 만큼 나누어 연산)
    input_splits = {}
    for k, v in kwargs.items():
        input_splits[k] = tf.split(v, num_gpus)

    output_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                output_split.append(fn(**{k: v[i] for k, v in input_splits.items()}))
    fn_output_len = len(output_split[0])
    output = [tf.reduce_mean(tf.stack([output_split[j][0] for j in range(num_gpus)], axis=0))] # loss
    for i in range(1, fn_output_len): # other outputs
        output.append(tf.concat([output_split[j][i] for j in range(num_gpus)], axis=0))
    return output

class NN:
    def __init__(self, name, char_dim):
        self.logger = get_logger(name)
        self.name = name
        self.char_dim = char_dim + 1
    
    def hidden_layer(self):
        def _hidden_layer(X_product_jaso, X_product_char, X_img_feat,
                          Y_bcate, Y_mcate, Y_scate, Y_dcate):
            
            # LAYER 1 - Text Layer 1: Jaso Embedding
            jaso_embedding = tf.get_variable('jaso_embedding',
                                             shape=[215, opt.jaso_embed_dim],
                                            )  # 215는 jaso_decomposer.py 참고
            jaso_embedded_X_list = []
            with tf.name_scope('jaso_embedded'):
                for X_text in [X_product_jaso]:
                    jaso_embedded_X_list.append(tf.nn.embedding_lookup(jaso_embedding, X_text))
            

            # LAYER 1 - Text Layer 1: Char Embedding
            char_embedding = tf.get_variable('char_embedding',
                                             shape=[self.char_dim, opt.char_embed_dim],
                                            )
            char_embedded_X_list = []
            with tf.name_scope('char_embedded'):
                for X_text in [X_product_char]:
                    char_embedded_X_list.append(tf.nn.embedding_lookup(char_embedding, X_text))

            
            # LAYER 1 - Text Layer 2: Bi-RNN (Jaso)
            jaso_rnn_output_list = []
            for i, embedded in enumerate(jaso_embedded_X_list):
                with tf.name_scope('jaso_rnn%d'%i):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(200)
                    bw_cell = tf.nn.rnn_cell.LSTMCell(200)

                    (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded, dtype=tf.float32,
                                                                                   scope='jaso_rnn%d'%i)
                    rnn_outputs = tf.concat([output_fw, output_bw], axis=2)
                    jaso_rnn_output_list.append(tf.expand_dims(rnn_outputs, -1))
            
            # LAYER 1 - Text Layer 3: CNN (Jaso)
            jaso_cnn_outputs = [] 
            for i, rnn_output in enumerate(jaso_rnn_output_list):
                with tf.name_scope('jaso_conv%d'%i):
                    for filter_size1 in [2, 3, 4, 5]:
                        with tf.name_scope('jaso_conv1_fs%d'%(filter_size1)):
                            conv1 = tf.layers.conv2d(rnn_output,
                                                    filters=128,
                                                    kernel_size=[filter_size1, 200*2],
                                                    strides=[1, 1],
                                                    activation=tf.nn.relu,)
                            pooled1 = tf.layers.max_pooling2d(conv1, [3, 1], strides=[2, 1],)
                            pooled1 = tf.transpose(pooled1, [0, 1, 3, 2])
                            for filter_size2 in [1, 2, 3]:
                                with tf.name_scope('jaso_conv2_fs%d'%(filter_size2)):
                                    # Convolution Layer
                                    conv2 = tf.layers.conv2d(pooled1,
                                                            filters=128,
                                                            kernel_size=[filter_size2, pooled1.shape[2]],
                                                            strides=[1, 1],
                                                            activation=tf.nn.relu, )
                                    # Maxpooling Layer
                                    pooled2 = tf.layers.max_pooling2d(conv2, [conv2.shape[1], 1], strides=[1, 1], )
                                    jaso_cnn_outputs.append(pooled2)
            
            # LAYER 1 - Text Layer 2: Bi-RNN (Char)
            char_rnn_output_list = []
            for i, embedded in enumerate(char_embedded_X_list):
                with tf.name_scope('char_rnn%d'%i):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(200)
                    bw_cell = tf.nn.rnn_cell.LSTMCell(200)

                    (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded, dtype=tf.float32,
                                                                                   scope='char_rnn%d'%i)
                    rnn_outputs = tf.concat([output_fw, output_bw], axis=2)
                    char_rnn_output_list.append(tf.expand_dims(rnn_outputs, -1))
            
            # LAYER 1 - Text Layer 3: CNN (Char)
            char_cnn_outputs = [] 
            for i, rnn_output in enumerate(char_rnn_output_list):
                with tf.name_scope('char_conv%d'%i):
                    for filter_size1 in [2, 3, 4, 5]:
                        with tf.name_scope('char_conv1_fs%d'%(filter_size1)):
                            conv1 = tf.layers.conv2d(rnn_output,
                                                    filters=128,
                                                    kernel_size=[filter_size1, 200*2],
                                                    strides=[1, 1],
                                                    activation=tf.nn.relu,)
                            pooled1 = tf.layers.max_pooling2d(conv1, [3, 1], strides=[2, 1],)
                            pooled1 = tf.transpose(pooled1, [0, 1, 3, 2])
                            for filter_size2 in [1, 2, 3]:
                                with tf.name_scope('char_conv2_fs%d'%(filter_size2)):
                                    # Convolution Layer
                                    conv2 = tf.layers.conv2d(pooled1,
                                                            filters=128,
                                                            kernel_size=[filter_size2, pooled1.shape[2]],
                                                            strides=[1, 1],
                                                            activation=tf.nn.relu, )
                                    # Maxpooling Layer
                                    pooled2 = tf.layers.max_pooling2d(conv2, [conv2.shape[1], 1], strides=[1, 1], )
                                    char_cnn_outputs.append(pooled2)


            cnn_concat = tf.concat(jaso_cnn_outputs + char_cnn_outputs, 3)
            cnn_concat = tf.reshape(cnn_concat, [-1, cnn_concat.shape[-1]])
            text_layer_output = tf.layers.dropout(cnn_concat, rate=1-opt.dropout_keep, training=self.is_training)
            
            # Concatenate all features
            #X_img_feat = tf.nn.softmax(X_img_feat)
            flat_input = tf.concat([text_layer_output, X_img_feat], axis=1)
            output = tf.contrib.layers.fully_connected(flat_input, opt.num_bcate+opt.num_mcate+opt.num_scate+opt.num_dcate+4, activation_fn=None)
            print(flat_input)
            output = tf.split(output, [opt.num_bcate+1, opt.num_mcate+1, opt.num_scate+1, opt.num_dcate+1], axis=1)
            print(output)
            # LAYER 2 - bcate_output
            bcate_output = output[0]
            
            # LAYER 2 - mcate_output
            mcate_output = output[1]
            
            # LAYER 2 - scate_output
            scate_output = output[2]
            
            # LAYER 2 - dcate_output
            dcate_output = output[3]

            # Softmax (for ensemble) 
            bcate_softmax = tf.nn.softmax(bcate_output)
            mcate_softmax = tf.nn.softmax(mcate_output)
            scate_softmax = tf.nn.softmax(scate_output)
            dcate_softmax = tf.nn.softmax(dcate_output)
            
            # Category Prediction
            bcate_pred = tf.argmax(bcate_output, axis=1)
            mcate_pred = tf.argmax(mcate_output, axis=1)
            scate_pred = tf.argmax(scate_output, axis=1)
            dcate_pred = tf.argmax(dcate_output, axis=1)

            # Loss
            nan_scate_mask = tf.not_equal(Y_scate, -1) 
            nan_dcate_mask = tf.not_equal(Y_dcate, -1)
            Y_bcate_onehot = tf.one_hot(Y_bcate, opt.num_bcate+1)
            Y_mcate_onehot = tf.one_hot(Y_mcate, opt.num_mcate+1)
            Y_scate_onehot = tf.one_hot(Y_scate, opt.num_scate+1)
            Y_dcate_onehot = tf.one_hot(Y_dcate, opt.num_dcate+1)
                    
            bcate_loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_bcate_onehot, logits=bcate_output))
            mcate_loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_mcate_onehot, logits=mcate_output))
            scate_loss = tf.reduce_mean(
                                tf.multiply(
                                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_scate_onehot, logits=scate_output),
                                    tf.cast(nan_scate_mask, tf.float32)))
            dcate_loss = tf.reduce_mean(
                                tf.multiply(
                                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_dcate_onehot, logits=dcate_output),
                                    tf.cast(nan_dcate_mask, tf.float32)))
            
            loss = bcate_loss + 1.2*mcate_loss + 1.3*scate_loss + 1.4*dcate_loss

            return loss, bcate_pred, mcate_pred, scate_pred, dcate_pred, bcate_softmax, mcate_softmax, scate_softmax, dcate_softmax
        return _hidden_layer
    
    def build_model(self, sess, use_n_gpus):
                
        self.sess = sess

        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_lr = tf.train.exponential_decay(opt.lr, self.global_step, opt.decay_step, opt.lr_decay, staircase=True)
        lr_limit = tf.constant(2e-4)
        self.decaying_lr = tf.cond(self.decaying_lr > lr_limit, lambda:self.decaying_lr , lambda:lr_limit)
        # Input Placeholders
        self.X_product_jaso = tf.placeholder(tf.int32, [None, opt.max_product_jaso_len], name='X_product_jaso')
        self.X_product_char = tf.placeholder(tf.int32, [None, opt.max_product_char_len], name='X_product_char')
        self.X_img_feat = tf.placeholder(tf.float32, [None, 2048], name='X_img_feat')
        
        # Output Placeholders
        self.Y_bcate = tf.placeholder(tf.int32, [None,])
        self.Y_mcate = tf.placeholder(tf.int32, [None,])
        self.Y_scate = tf.placeholder(tf.int32, [None,])
        self.Y_dcate = tf.placeholder(tf.int32, [None,])

        # Hidden layer - Multi-GPUS
        outputs = make_parallel(self.hidden_layer(), use_n_gpus,
                                X_product_jaso=self.X_product_jaso, X_product_char=self.X_product_char,
                                X_img_feat=self.X_img_feat,
                                Y_bcate=self.Y_bcate, Y_mcate=self.Y_mcate,
                                Y_scate=self.Y_scate, Y_dcate=self.Y_dcate)
        self.loss, self.bcate_pred, self.mcate_pred, self.scate_pred, self.dcate_pred, \
                self.bcate_softmax, self.mcate_softmax, self.scate_softmax, self.dcate_softmax = outputs
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.decaying_lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, colocate_gradients_with_ops=True)
   
        self.logger.info("# Trainable Parameters: %d"%np.sum(
                [np.prod(v.shape) for v in tf.trainable_variables()]))

    @staticmethod
    def calc_acc(reals, preds):
        not_nan = np.not_equal(reals, -1)
        correct = np.equal(reals[not_nan], preds[not_nan])
        return sum(correct) / sum(not_nan)
        
    @staticmethod
    def calc_correct(reals, preds):
        not_nan = np.not_equal(reals, -1)
        correct = np.equal(reals[not_nan], preds[not_nan])
        return sum(correct), sum(not_nan)
    
    def fit(self, train_gen, train_steps, val_gen, val_steps, epochs, resume):
        
        epoch_from = 1
        if resume !=0 :
            epoch_from = self.load(resume) + 1
        epoch_end = epoch_from + epochs
        status = {}
        for epoch in range(epoch_from, epoch_end):
            self.logger.info(f"[EPOCH: {epoch}]")
            # Training...
            correct = {'b':0, 'm':0, 's':0, 'd':0}
            total = {'b':0, 'm':0, 's':0, 'd':0}
            acc = {'b':0, 'm':0, 's':0, 'd':0} 
            pred = {'b':0, 'm':0, 's':0, 'd':0} 
            loss, score = 0, 0
            pbar = tqdm.tqdm(range(1, train_steps+1))
            for step in pbar:
                X, Y = next(train_gen)
                run_tensors = [self.train_op, self.loss, 
                               self.bcate_pred, self.mcate_pred, self.scate_pred, self.dcate_pred]
                feed_dict = {self.X_product_jaso: X['product_jaso'], self.X_product_char: X['product_char'], 
                             self.X_img_feat: X['img_feat'],
                             self.Y_bcate: Y['bcateid'], self.Y_mcate: Y['mcateid'],
                             self.Y_scate: Y['scateid'], self.Y_dcate: Y['dcateid'],
                             self.is_training:True}
                _, l, pred['b'], pred['m'], pred['s'], pred['d'] =  self.sess.run(run_tensors, feed_dict=feed_dict)
                
                loss += l
                for cate in ['b', 'm', 's', 'd']:
                    c, t = self.calc_correct(Y[cate+'cateid'], pred[cate])
                    correct[cate] += c
                    total[cate] += t
                    acc[cate] = correct[cate] / total[cate]
                score = (acc['b'] + 1.2*acc['m'] + 1.3*acc['s'] + 1.4*acc['d']) / 4
                pbar.set_description('[Train] loss:{:.5f} b:{:.4f} m:{:.4f} s:{:.4f} d:{:.4f} score:{:.4f}'.format(
                            loss/step, acc['b'], acc['m'], acc['s'], acc['d'], score))
                
                # mini validation
                if step % 300 == 0 and val_steps != 0:
                    _correct = {'b':0, 'm':0, 's':0, 'd':0}
                    _total = {'b':0, 'm':0, 's':0, 'd':0}
                    _acc = {'b':0, 'm':0, 's':0, 'd':0} 
                    _pred = {'b':0, 'm':0, 's':0, 'd':0} 

                    X, Y = next(val_gen)
                    run_tensors = [self.loss,
                                   self.bcate_pred, self.mcate_pred, self.scate_pred, self.dcate_pred]
                    feed_dict = {self.X_product_jaso: X['product_jaso'], self.X_product_char: X['product_char'], 
                                 self.X_img_feat: X['img_feat'],
                                 self.Y_bcate: Y['bcateid'], self.Y_mcate: Y['mcateid'],
                                 self.Y_scate: Y['scateid'], self.Y_dcate: Y['dcateid'],
                                 self.is_training:False}
                    _l, _pred['b'], _pred['m'], _pred['s'], _pred['d'] =  self.sess.run(run_tensors, feed_dict=feed_dict)
                    
                    for cate in ['b', 'm', 's', 'd']:
                        c, t = self.calc_correct(Y[cate+'cateid'], _pred[cate])
                        _correct[cate] += c
                        _total[cate] += t
                        _acc[cate] = _correct[cate] / _total[cate]
                    _score = (_acc['b'] + 1.2*_acc['m'] + 1.3*_acc['s'] + 1.4*_acc['d']) / 4
                    self.logger.info('\n[miniVal] loss:{:.5f} b:{:.4f} m:{:.4f} s:{:.4f} d:{:.4f} score:{:.4f}'.format(
                                _l, _acc['b'], _acc['m'], _acc['s'], _acc['d'], _score))
     

            # Train log
            log = '\n- loss: {:.5f} - acc(b): {:.4f} - acc(m): {:.4f} - acc(s): {:.4f} - acc(d): {:.4f} - score: {:.4f}'.format(
                        loss/train_steps, acc['b'], acc['m'], acc['s'], acc['d'], score)
            self.logger.info(log) 
            status['t_loss']=loss/train_steps
            status['t_acc_b']=acc['b']
            status['t_acc_m']=acc['m']
            status['t_acc_s']=acc['s']
            status['t_acc_d']=acc['d']
            status['t_score']=score

            # Validation...
            correct = {'b':0, 'm':0, 's':0, 'd':0}
            total = {'b':0, 'm':0, 's':0, 'd':0}
            acc = {'b':0, 'm':0, 's':0, 'd':0} 
            pred = {'b':0, 'm':0, 's':0, 'd':0} 
            loss, score = 0, 0
            pbar = tqdm.tqdm(range(1, val_steps+1))
            for step in pbar:
                X, Y = next(val_gen)
                run_tensors = [self.loss,
                               self.bcate_pred, self.mcate_pred, self.scate_pred, self.dcate_pred]
                feed_dict = {self.X_product_jaso: X['product_jaso'], self.X_product_char: X['product_char'], 
                             self.X_img_feat: X['img_feat'],
                             self.Y_bcate: Y['bcateid'], self.Y_mcate: Y['mcateid'],
                             self.Y_scate: Y['scateid'], self.Y_dcate: Y['dcateid'],
                             self.is_training:False}
                
                l, pred['b'], pred['m'], pred['s'], pred['d'] =  self.sess.run(run_tensors, feed_dict=feed_dict)
                
                loss += l
                for cate in ['b', 'm', 's', 'd']:
                    c, t = self.calc_correct(Y[cate+'cateid'], pred[cate])
                    correct[cate] += c
                    total[cate] += t
                    acc[cate] = correct[cate] / total[cate]
                score = (acc['b'] + 1.2*acc['m'] + 1.3*acc['s'] + 1.4*acc['d']) / 4
                pbar.set_description('[Val] loss:{:.5f} b:{:.4f} m:{:.4f} s:{:.4f} d:{:.4f} score:{:.4f}'.format(
                            loss/step, acc['b'], acc['m'], acc['s'], acc['d'], score))
            # Validation log
            if val_steps == 0:
                status['v_loss']=0
            else:
                log = '\n- loss: {:.5f} - acc(b): {:.4f} - acc(m): {:.4f} - acc(s): {:.4f} - acc(d): {:.4f} - score: {:.4f}'.format(
                            loss/val_steps, acc['b'], acc['m'], acc['s'], acc['d'], score)
                self.logger.info(log)
                status['v_loss']=loss/val_steps
                
            status['v_acc_b']=acc['b']
            status['v_acc_m']=acc['m']
            status['v_acc_s']=acc['s']
            status['v_acc_d']=acc['d']
            status['v_score']=score
            
            status['lr'] = float(self.decaying_lr.eval())
            status['batch_size'] = opt.batch_size

            # Save model and status(log)
            self.save(epoch, status)

    def predict(self, data_gen, total_steps, softmax=False):
        """
        주의: 현재 모델이 multi-gpu에 정의되어있다면, batch split시 오류 발생할 수 있음.
        walk-around:  classifier.py->predict를 통해 이 함수가 호출되면 문제 없음
        """
        y_preds = {'b':[], 'm':[], 's':[], 'd':[]}
        pred = {}
        pbar = tqdm.tqdm(range(1, total_steps+1))
        for step in pbar:
            X = next(data_gen)
            if softmax==False:
                run_tensors = [self.bcate_pred, self.mcate_pred, self.scate_pred, self.dcate_pred]
            else:
                run_tensors = [self.bcate_softmax, self.mcate_softmax, self.scate_softmax, self.dcate_softmax]

            feed_dict = {self.X_product_jaso: X['product_jaso'], self.X_product_char: X['product_char'], 
                         self.X_img_feat: X['img_feat'],
                         self.is_training:False}
            pred['b'], pred['m'], pred['s'], pred['d'] =  self.sess.run(run_tensors, feed_dict=feed_dict)
            for cate in pred:
                y_preds[cate].extend(pred[cate])
        return y_preds

    def save(self, epoch, status):
        # Model save
        save_meta = False
        if epoch == 1:
            save_meta = True
            os.makedirs(os.path.join('model', self.name), exist_ok=True)
        ckpt_path = os.path.join('model', self.name, 'model_{}.ckpt'.format(epoch))
        saver = tf.train.Saver()
        saved_path = saver.save(self.sess, ckpt_path, write_meta_graph=save_meta)
        self.logger.info('Model Saved! {}'.format(saved_path))

        # Status(log) save
        status['time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        status_path = os.path.join('model', self.name, 'status_{}.json'.format(epoch))
        with open(status_path, 'w') as f:
            json.dump(status, f, sort_keys=True, indent=4)
        
    def load(self, epoch=-1):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.join('model', self.name))
        if ckpt is None:
            self.logger.info('There is no checkpoint')
            return 0
        ckpt_path = ckpt.model_checkpoint_path
        if epoch != -1:
            ckpt_path = re.sub('\d+(?=\.ckpt)', str(epoch), ckpt_path)

        saver.restore(self.sess, ckpt_path)
        self.logger.info('Model loaded! {}'.format(ckpt_path))

        self.logger.info('EPOCH {}'.format(epoch))
        return int(re.findall('\d+(?=\.ckpt)', ckpt_path)[0])
    
