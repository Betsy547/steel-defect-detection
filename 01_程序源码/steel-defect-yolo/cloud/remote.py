#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过密码登录 AutoDL 实例执行命令 / 传文件（本地辅助脚本）。

密码从环境变量读取，不写进代码：

    export SSH_HOST=connect.nmb1.seetacloud.com
    export SSH_PORT=13731
    export SSH_PASS='******'

用法：
    python cloud/remote.py run "nvidia-smi"
    python cloud/remote.py put data.yaml /root/autodl-tmp/steel/data.yaml
    python cloud/remote.py get /root/autodl-tmp/steel/out/best.pt D:/yolo/weights/best.pt
"""

import os
import sys
from pathlib import Path

import paramiko

# 远端输出里有 emoji / 中文，Windows 控制台默认 GBK 会崩
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HOST = os.environ.get("SSH_HOST", "")
PORT = int(os.environ.get("SSH_PORT", "22"))
USER = os.environ.get("SSH_USER", "root")
PASS = os.environ.get("SSH_PASS", "")


def connect() -> paramiko.SSHClient:
    if not HOST or not PASS:
        sys.exit("请先设置 SSH_HOST / SSH_PORT / SSH_PASS 环境变量")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS,
                   timeout=20, banner_timeout=20, auth_timeout=20)
    return client


def run(client, command: str, timeout: float = 600) -> int:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout,
                                                get_pty=False)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    err = stderr.read().decode("utf-8", "replace")
    if err.strip():
        sys.stderr.write(err)
    return stdout.channel.recv_exit_status()


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    action = sys.argv[1]
    client = connect()
    try:
        if action == "run":
            return run(client, sys.argv[2])
        if action == "put":
            local, remote = Path(sys.argv[2]), sys.argv[3]
            run(client, f"mkdir -p {os.path.dirname(remote) or '.'}")
            sftp = client.open_sftp()
            try:
                sftp.put(str(local), remote)
            finally:
                sftp.close()
            print(f"已上传 {local} -> {remote}")
            return 0
        if action == "get":
            remote, local = sys.argv[2], Path(sys.argv[3])
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            try:
                sftp.get(remote, str(local))
            finally:
                sftp.close()
            print(f"已下载 {remote} -> {local}")
            return 0
    finally:
        client.close()

    sys.exit(f"未知操作：{action}")


if __name__ == "__main__":
    raise SystemExit(main())
