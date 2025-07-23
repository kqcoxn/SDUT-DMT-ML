"""
通用函数
"""

import json


class PresetsManager:
    """
    预设管理器
    """

    def __init__(self):
        # 文件路径
        self.file_path = "presets.json"

        # 读取已有配置
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.presets = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.presets = {}

    def set(self, key: str, value: str):
        """
        设置键值对
        :param key: 键
        :param value: 值
        """
        self.presets[key] = value
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.presets, f)


presets_manager = PresetsManager()
