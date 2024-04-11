import yaml
import json


def read_yaml(yaml_file: str):
    with open(yaml_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        return data


def read_json(file):
    """
    :param file:    待读取的json文件
    :return:
    """
    with open(file, 'r') as f:
        data = json.load(f)
        f.close()
    return data


def write_json(content, file_name):
    with open(file_name, 'w') as file:
        json.dump(content, file)
        file.close()