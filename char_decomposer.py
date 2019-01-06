
import os
import numpy as np
import pickle
from misc import Option
opt = Option('./config.json')

char_dict = {}

def str_to_chars_train(string):
    char_list = []
    for character in string:
        c = char_dict.get(character, 0)

        # dict에 존재하지 않으면 새로 번호 할당
        if c == 0:
            c = len(char_dict) + 1
            char_dict[character] = c
            print('new_character', character, c)
        char_list.append(c)

    return char_list

def str_to_chars_test(string):
    char_list = []
    for character in string:
        c = char_dict.get(character, 0)
        char_list.append(c)

    return char_list

def decompose_as_chars(data: list, max_str_len: int, is_train: bool):
    global char_dict
    if is_train:
        if os.path.exists(os.path.join(opt.data_path, 'char_dict.pkl')):
            char_dict = pickle.load(open(os.path.join(opt.data_path, 'char_dict.pkl'), 'rb'), encoding='utf-8')
        
        decomposed_data = [str_to_chars_train(string.decode()) for string in data]
    else:
        char_dict = pickle.load(open(os.path.join(opt.data_path, 'char_dict.pkl'), 'rb'), encoding='utf-8')
        decomposed_data = [str_to_chars_test(string.decode()) for string in data]

    result = np.zeros((len(data), max_str_len), dtype=np.int32)
    for i, seq in enumerate(decomposed_data):
        length = len(seq)
        if length >= max_str_len:
            length = max_str_len
        result[i, :length] = np.array(seq)[:length]
    
    # char_dict 저장
    if is_train:
        with open(os.path.join(opt.data_path, 'char_dict.pkl'), 'wb') as fout:
            pickle.dump(char_dict, fout)
    
    return result

