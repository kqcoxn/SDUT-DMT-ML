"""
参数解析器
"""

import argparse


def parse_args():
    """
    参数解析
    - 命令类型
        - 添加键值对
            - 键值对
        - 修改文件名
            - 修改方式
                - 单文件
                - 目录所有文件
                - 仅输出
            - 目标路径
            - 修改内容
    """
    parser = argparse.ArgumentParser(description="文件重命名助手")
    parser.add_argument("command", choices=["set", "s", "rename", "r"], help="操作类型")
    # 添加键值对
    parser.add_argument("-kv", metavar="key=value", help="添加的键值对")
    # 修改文件名
    parser.add_argument("--dir", "-d", dest="dir_path", help="目标文件夹路径")
    parser.add_argument("--file", "-f", dest="file_path", help="目标文件路径")
    parser.add_argument("--template", "-t", help="修改模板")

    args = parser.parse_args()
    return args
