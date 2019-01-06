# -*- coding: utf-8 -*-
import os
import numpy as np

cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
jong = ['', 'ㄱ', 'ㄲ', 'ㄱㅅ', 'ㄴ', 'ㄴㅈ', 'ㄴㅎ', 'ㄷ', 'ㄹ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ', 'ㅁ',
        'ㅂ', 'ㅂㅅ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'] # len = 27 (+1)

korean_len = len(cho) + len(jung) + len(jong) - 1  # == 67

def char_to_jasos(character):
    # [0,66]: 한글 67개 / [67,162]: ASCII 96개 / [163,213] : 한글 단자음, 단모음 51개 -> 총 214가지
    result = []
    uni = ord(character)    

    # character가 한글인 경우
    if 44032 <= uni <= 55203:  # 가:44032 ~ 힣: 55203
        x = uni - 44032  # character - ord('가')
        y = x // 28
        z = x % 28
        x = y // 21
        y = y % 21
        # zz = jong[z]

        result.append(x)
        result.append(len(cho) + y)
        # 종성이 존재하는 경우
        if z > 0:
            result.append(len(cho) + len(jung) + z - 1)
        return result
    else:
        # ASCII(non-printing character 32개 제외) , 67~
        if 32 <= uni < 128:
            result = 35 + uni # korean_len + uni - 32
        # 단자음, 단모음, [ㄱ:12593]~[ㅣ:12643], 194~
        elif ord('ㄱ') <= uni <= ord('ㅣ'):
            result = uni - 12430  # korean_len + 96 + (uni - 12593)
        # undefined character
        else:
            result = 214  # korean_len + 128 + 51

        return [result]

def str_to_jasos(string):
    jaso_list = []
    for character in string:
        char_as_jasos = char_to_jasos(character)
        jaso_list.extend(char_as_jasos)
    return jaso_list

def decompose_as_jasos(data: list, max_str_len: int):
    """
    max_str_len 보다 긴 string에 대하여, 초과한 부분은 버린다.
    max_str_len 보다 짧은 string에 대하여, 빈 공간은 0으로 채운다.
    :param data: list of string, string이 byte데이터로 존재하여야 함. 그렇지 않은경우 .decode() 지우면 됨
    :param max_str_len: maximum length of string. 
    :return: e.g. [[0, 1, 5, 6], [5, 4, 10, 200], ...] max_str_len이 4일때
    """
    decomposed_data = [str_to_jasos(string.decode()) for string in data]
    result = np.zeros((len(data), max_str_len), dtype=np.int32)
    for i, seq in enumerate(decomposed_data):
        length = len(seq)
        if length >= max_str_len:
            length = max_str_len
        result[i, :length] = np.array(seq)[:length]

    return result
