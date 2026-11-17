#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Windows 控制台中文输出兼容。

Windows 上 Python 默认按系统代码页（cp936）写 stdout，在 Git Bash /
Windows Terminal / VS Code 里会显示成乱码。导入本模块后自动切到 UTF-8。

用法：在所有会打印中文的脚本里加一行
    import _compat  # noqa: F401
"""

import sys

def _setup():
    try:
        for stream in (sys.stdout, sys.stderr):
            enc = (getattr(stream, "encoding", "") or "").lower()
            if enc and enc.replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8")
    except Exception:
        # 某些环境（如被重定向到不支持 reconfigure 的流）忽略即可
        pass


_setup()
