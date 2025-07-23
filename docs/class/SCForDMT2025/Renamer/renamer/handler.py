"""
命令匹配
"""

import argparse

from .resolver import rename_file, rename_dir, get_result, set_kv


def handle_args(args: argparse.Namespace):
    """
    命令匹配
    :param args: 行参数
    """
    command = args.command
    if command == "rename" or command == "r":
        template = args.template
        if args.file_path:
            rename_file(args.file_path, template)
        elif args.dir_path:
            rename_dir(args.dir_path, template)
        else:
            result = get_result(template)
            print(result)
    elif command == "set" or command == "s":
        set_kv(args.kv)
    else:
        raise ValueError("指令错误")
