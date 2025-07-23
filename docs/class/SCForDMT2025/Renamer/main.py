"""
在入口文件中，程序应只包含功能索引的代码，具体实现逻辑应封装在其他文件中
"""

from parser import parse_args
from renamer.handler import handle_args


def main():
    """
    主程序
    """
    # 解析输入参数
    args = parse_args()

    # 根据参数处理文件
    handle_args(args)


if __name__ == "__main__":
    """
    程序入口
    """
    main()
