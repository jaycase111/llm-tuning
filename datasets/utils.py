import json
from typing import List

def read_json(file):
    """
    :param file:    待读取的json文件
    :return:
    """
    with open(file, 'r') as f:
        data = json.load(f)
        f.close()
    return data


def write_json(file: str,
               content: object):
    """
    :param file:        待写入的json文件
    :param content:     待写入的对象
    :return:
    """
    with open(file, 'w') as f:
        json.dump(content, f)
        f.close()


def get_pair_element(object_list: List[object]):
    """
    :param object_list:  长度为偶数的列表
    :return:    将相邻两个元素防在同一个列表的双层列表
    """
    assert len(object_list) > 0 and len(object_list) % 2 == 0
    pair_object_list = []
    for i in range(len(object_list) // 2):
        pair_object_list.append([object_list[i * 2], object_list[i * 2 + 1]])
    return pair_object_list
