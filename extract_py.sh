#!/bin/bash

# 使用 find 命令递归查找当前目录下所有 .py 文件
# -type f 表示只找文件
# -name "*.py" 匹配 Python 后缀
find . -type f -name "*.py" | while read -r file; do
    echo "========================================"
    echo "PATH: $file"
    echo "========================================"
    cat "$file"
    echo -e "\n" # 在文件末尾加个换行，方便区分
done
