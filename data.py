import os
# os.environ['OMP_NUM_THREADS'] = '1'
import sys

import tqdm
import fire
import h5py
import numpy as np
import pickle

from misc import get_logger, Option
from jaso_decomposer import decompose_as_jasos
from char_decomposer import decompose_as_chars



opt = Option('./config.json')

class Rawdata:

    def __init__(self):
        self.logger = get_logger('rawdata')

    def make_h5(self, div):
        path_list = {'train': opt.train_data_list,
                     'dev': opt.dev_data_list, 
                     'test': opt.test_data_list}
        h5_chunk_size = 256  # h5 내부적 접근 단위
        total_data_len = 0
        for chunk in path_list[div]:
            chunk_len = h5py.File(chunk, 'r')[div]['pid'].shape[0]
            total_data_len += chunk_len
        self.logger.info('Total data length(%s): %d' %(div, total_data_len))
        
        os.makedirs(opt.data_path, exist_ok=True)
        with h5py.File(os.path.join(opt.data_path, 'raw_%s.h5'%div), 'w') as fout:
            # create dataset
            chunk_sample = h5py.File(path_list[div][0], 'r')
            for col in chunk_sample[div].keys():
                col_shape = chunk_sample[div][col].shape
                col_dtype = chunk_sample[div][col].dtype
                if col == 'img_feat':
                    fout.create_dataset(col, shape=(total_data_len, 2048),
                        dtype=col_dtype, chunks=(h5_chunk_size, 2048))
                else:
                    fout.create_dataset(col, shape=(total_data_len,),
                        dtype=col_dtype, chunks=(h5_chunk_size, ))
            
            # copy chunk
            idx_from = 0
            for chunk_path in path_list[div]:
                self.logger.info('processing... %s' % chunk_path)
                h_chunk = h5py.File(chunk_path, 'r')
                size = h_chunk[div]['pid'].shape[0]
                for col in h_chunk[div].keys():
                    self.logger.info('merging... %s' % col)
                    fout[col][idx_from : idx_from+size] = h_chunk[div][col][:]
                idx_from += size
        
        self.logger.info('Done!')


class Data:

    def __init__(self):
        self.logger = get_logger('data')
    
    def preprocessing(self, div):
        raw_h5 = h5py.File(os.path.join(opt.data_path, 'raw_%s.h5'%div) ,'r')
        with h5py.File(os.path.join(opt.data_path, '%s.h5'%div), 'w') as data_h5:
            # create dataset
            total_data_len = raw_h5['pid'].shape[0]
            h5_chunk_size = 256  # h5 내부적 접근 단위

            col_dtype = raw_h5['pid'].dtype
            data_h5.create_dataset('pid', shape=(total_data_len, ), 
                                  dtype=col_dtype, chunks=(h5_chunk_size, ))
            data_h5.create_dataset('price', shape=(total_data_len,), 
                                  dtype=int, chunks=(h5_chunk_size,))
            data_h5.create_dataset('img_feat', shape=(total_data_len, 2048), 
                                  dtype=np.float32, chunks=(h5_chunk_size, 2048))      
            
            if div == 'train':
                data_h5.create_dataset('bcateid', shape=(total_data_len,), 
                                      dtype=int, chunks=(h5_chunk_size,))
                data_h5.create_dataset('mcateid', shape=(total_data_len,), 
                                      dtype=int, chunks=(h5_chunk_size,))
                data_h5.create_dataset('scateid', shape=(total_data_len,), 
                                      dtype=int, chunks=(h5_chunk_size,))
                data_h5.create_dataset('dcateid', shape=(total_data_len,), 
                                      dtype=int, chunks=(h5_chunk_size,))

            data_h5.create_dataset('product_jaso', shape=(total_data_len, opt.max_product_jaso_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_product_jaso_len))
            data_h5.create_dataset('brand_jaso', shape=(total_data_len, opt.max_brand_jaso_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_brand_jaso_len))
            data_h5.create_dataset('model_jaso', shape=(total_data_len, opt.max_model_jaso_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_model_jaso_len))
            data_h5.create_dataset('maker_jaso', shape=(total_data_len, opt.max_maker_jaso_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_maker_jaso_len))

            data_h5.create_dataset('product_char', shape=(total_data_len, opt.max_product_char_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_product_char_len))
            data_h5.create_dataset('brand_char', shape=(total_data_len, opt.max_brand_char_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_brand_char_len))
            data_h5.create_dataset('model_char', shape=(total_data_len, opt.max_model_char_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_model_char_len))
            data_h5.create_dataset('maker_char', shape=(total_data_len, opt.max_maker_char_len), 
                                  dtype=int, chunks=(h5_chunk_size, opt.max_maker_char_len))
            
            # copy unchanged data
            self.logger.info('copy pid ...')
            data_h5['pid'][:] = raw_h5['pid'][:]
            self.logger.info('copy price ...')
            data_h5['price'][:] = raw_h5['price'][:]

            # text data parsing
            self.logger.info('parsing product ...')                            
            data_h5['product_jaso'][:] = decompose_as_jasos(raw_h5['product'][:], opt.max_product_jaso_len)
            data_h5['product_char'][:] = decompose_as_chars(raw_h5['product'][:], opt.max_product_char_len, div=='train')
            self.logger.info('parsing brand ...')          
            data_h5['brand_jaso'][:] = decompose_as_jasos(raw_h5['brand'][:], opt.max_brand_jaso_len)
            data_h5['brand_char'][:] = decompose_as_chars(raw_h5['brand'][:], opt.max_brand_char_len, div=='train')
            self.logger.info('parsing model ...')
            data_h5['model_jaso'][:] = decompose_as_jasos(raw_h5['model'][:], opt.max_model_jaso_len)
            data_h5['model_char'][:] = decompose_as_chars(raw_h5['model'][:], opt.max_model_char_len, div=='train')
            self.logger.info('parsing maker ...')
            data_h5['maker_jaso'][:] = decompose_as_jasos(raw_h5['maker'][:], opt.max_maker_jaso_len)
            data_h5['maker_char'][:] = decompose_as_chars(raw_h5['maker'][:], opt.max_maker_char_len, div=='train')
            
            # copy img_feat
            self.logger.info('copy img_feat ...')
            data_h5['img_feat'][:] = raw_h5['img_feat'][:]
            
            if div == 'train': 
                self.logger.info('copy cateid ...')
                data_h5['bcateid'][:] = raw_h5['bcateid'][:]
                data_h5['mcateid'][:] = raw_h5['mcateid'][:]
                data_h5['scateid'][:] = raw_h5['scateid'][:]
                data_h5['dcateid'][:] = raw_h5['dcateid'][:]
                
            self.logger.info('Done!')

if __name__ == '__main__':
    rawdata = Rawdata()
    data = Data()
    fire.Fire({'make_h5': rawdata.make_h5,
              'preprocessing': data.preprocessing})
