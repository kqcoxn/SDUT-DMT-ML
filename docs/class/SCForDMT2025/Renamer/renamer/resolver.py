"""
执行函数
"""

import json
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import presets_manager
from renamer.key_words import key_words_handler


def set_kv(kv: str):
    """
    设置键值对
    :param kv: 键值对key=value
    """
    # 获取参数
    try:
        key, value = kv.split("=", 1)
    except ValueError:
        raise ValueError("键值对格式错误，应为 'key=value'")

    if not key or not value:
        raise ValueError("键值对的键和值不能为空")

    # 向文件写入键值对
    presets_manager.set(key, value)
    print(f"成功保存预设{kv}")


def get_result(template: str, file_path: str = None) -> str:
    """
    处理结果
    :param template: 模板
    :param file_path: 文件路径
    :return: 处理结果
    """
    try:
        result = key_words_handler(template, file_path)
        result = result.format(**presets_manager.presets)
        if file_path:
            file_ext = os.path.splitext(file_path)[1]
            result += file_ext
        return result
    except KeyError as e:
        raise KeyError(f"键值对错误，缺少键{e.args[0]}")
    except AttributeError:
        raise AttributeError("未输入模板")
    except Exception as e:
        raise e


def rename_file(file_path: str, template: str):
    """
    重命名文件
    :param file_path: 文件路径
    :param template: 模板
    """
    try:
        result = get_result(template, file_path)
        dir_name = os.path.dirname(file_path)
        new_path = os.path.join(dir_name, result)
        os.rename(file_path, new_path)
        print(f"重命名成功：{file_path} -> {new_path}")
    except Exception as e:
        raise Exception(f"文件重命名失败，原因：{e.args[0]}")


def rename_dir(dir_path: str, template: str):
    """
    重命名文件夹
    :param dir_path: 文件夹路径
    :param template: 模板
    """
    try:
        files = os.listdir(dir_path)
        # 风险评估
        exts = []
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in exts:
                raise Exception(
                    f"重命名失败：目录下有多个同后缀（{ext}）文件，重命名后会重名，请检查模板或文件夹内容。"
                )
            exts.append(ext)

        # 检查通过后统一重命名
        for file in files:
            file_path = os.path.join(dir_path, file)
            result = get_result(template, file_path)
            new_path = os.path.join(dir_path, result)
            os.rename(file_path, new_path)
            print(f"重命名成功：{file_path} -> {new_path}")
    except Exception as e:
        raise Exception(f"重命名失败，原因：{e.args[0]}")
