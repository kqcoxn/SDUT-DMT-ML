import os

key_words = ["{origin}"]


def key_words_handler(template: str, file_path: str = None) -> str:
    """
    处理模板中的关键字
    :param template: 模板
    :param file_path: 文件路径
    :return: 处理结果
    """
    try:
        for key_word in key_words:
            if key_word in template:
                template = match_key_word(key_word, template, file_path)
        return template
    except Exception as e:
        print(f"重命名失败：{e}")


def match_key_word(key_word: str, template: str, file_path: str = None) -> str:
    """
    匹配关键字
    :param key_word: 关键字
    :param template: 模板
    :param file_path: 文件路径
    :return: 处理结果
    """
    if key_word == "{origin}":
        template = template.replace(key_word, get_origin_filename(file_path))
    return template


def get_origin_filename(file_path: str) -> str:
    """
    获取原始文件名
    :param file_path: 文件路径
    :return: 原始文件名
    """
    return os.path.splitext(os.path.basename(file_path))[0]
