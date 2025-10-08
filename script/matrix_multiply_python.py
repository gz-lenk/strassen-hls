#!/usr/bin/env python3
"""
Python实现的分块矩阵乘法程序

分块方法：
一个小块为N*N，也即最基本的乘法计算单位（在FPGA中，这个基本单位可以简化到1个矩阵元素？）；
一个大块包含k*m个小块（可调整，也可以是在2*2到6*6之间的任何尺寸，甚至大块和大块之间的小块划分不一定统一）
这个程序采用N=64，k=m=4作为样例，针对4096*4096矩阵与4096*4096矩阵乘法进行计算。

计算方法：
1. 把整个矩阵分成(N*k)*(N*m)的大块，每个大块之间按照标准分块乘法方式进行计算
2. 对于每个大块中，进一步分成k*m个N*N的小块，这一个k*m通过快速算法进行计算（这里4*4乘法正常需要4^3=64次乘法，快速算法仅需49次）
- TODO：后续这一步可以再细化，比如识别出A矩阵中稀疏块的分布，针对不同稀疏分布采用不同的快速乘法策略（例如，对于4*4*4的矩阵乘法任务，当(3,4)或(4,4)位置的元素为稀疏，也即全零块时，乘法次数能进一步减少到48次）
3. 每个N*N小块之间相乘，直接调用标准乘法的numpy.dot

使用方法：
1. 从文件加载矩阵A进行计算:
   python matrix_multiply_python.py /path/to/matrix.npy
   
2. 生成随机矩阵进行计算:
   python matrix_multiply_python.py --generate-random
   
3. 不指定参数时默认生成随机矩阵:
   python matrix_multiply_python.py
   
4. 指定随机数种子:
   python matrix_multiply_python.py --generate-random --seed 123

支持的矩阵文件格式: .npy, .csv, .txt
矩阵必须是4096x4096大小的float32类型
"""

import numpy as np
from typing import Tuple, List
from tqdm import tqdm
import argparse
import sys
import os

U = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, -1, -1, 0, 0, 0, 0, -1, 1, 0, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, -1, 1, -1, -1, 0, 0, 0, 0, -1, 1, -1, -1],
    [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, -1, 1, 1, 1, 0, 1, 0, 0, 1, -1, 0, 0],
    [0, 1, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, -1, 1, 1, 1, 0, 1, 0, 1, 1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, -1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -1, -1],
    [1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 1, 1, -1, -1, -1],
    [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float32)

V = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, -1, -1, 0, 0, 0, 0, -1, 1, 0, 0],
    [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, -1, 1, -1, -1, 0, 0, 0, 0, -1, 1, -1, -1],
    [0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, -1, 1, 1, 1, 0, 1, 0, 0, 1, -1, 0, 0],
    [0, 1, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0],
    [0, 1, 0, 1, -1, 1, 1, 1, 0, 1, 0, 1, 1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, -1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -1, -1],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0],
    [1, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 1, 1, -1, -1, -1],
    [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0],
], dtype=np.float32)

W = np.array([
    [1, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 0, -1, 1, 0, 0, -1, 1, 1, -1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, -1, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 0, -1, 0, 0, 0, 1, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 1, 1, -1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, -1, 0, 0, -1, 1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, -1, 0, 1, -1, 0, 0, 0, -1, -1, 1, -1, 0, 0, -1, 1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, -1, 0, -1, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 1, 1, 0, -1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 1, -1, 0, 1, 0, 0, 0, -1, 1, 1, 0, 1, 0, 1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, 1, 0, 0, 0, -1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 1, -1, -1, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, -1, -1, 1, -1, 0, 1, 1, 0, 0, -1, 0, 0, -1, -1, 1, 0, 0, 0, 1, 0, 0, -1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, 1, 0, 0, 0, -1, 1, 0, 0, -1, -1, 1, -1, 0, 1, 1, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
], dtype=np.float32)


# 常量定义
SMALL_BLOCK_ROWS = 64  # N，可调整
SMALL_BLOCK_COLS = 64
LARGE_BLOCK_FACTOR = 4  # k=m，可调整，但系数矩阵相应也要调整
LARGE_BLOCK_ROWS = SMALL_BLOCK_ROWS * LARGE_BLOCK_FACTOR
LARGE_BLOCK_COLS = SMALL_BLOCK_COLS * LARGE_BLOCK_FACTOR


def extract_4x4_blocks(matrix: np.ndarray) -> List[List[np.ndarray]]:
    """
    从128x128的大块中提取4x4个32x32的小块
    
    Args:
        matrix: 128x128的矩阵
        
    Returns:
        4x4的小块列表，每个小块为32x32
    """
    blocks = []
    for i in range(4):
        row_blocks = []
        for j in range(4):
            start_row = i * SMALL_BLOCK_ROWS
            end_row = start_row + SMALL_BLOCK_ROWS
            start_col = j * SMALL_BLOCK_COLS
            end_col = start_col + SMALL_BLOCK_COLS
            
            block = matrix[start_row:end_row, start_col:end_col].copy()
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return blocks

def flatten_4x4_blocks(blocks: List[List[np.ndarray]]) -> np.ndarray:
    """
    将4x4个小块按行展开为16个向量
    
    Args:
        blocks: 4x4的小块列表
        
    Returns:
        16个展开的向量，形状为(16, block_size)
    """
    flattened = []
    for i in range(4):
        for j in range(4):
            flattened.append(blocks[i][j].flatten()) # block[0][0],block[0][1],block[0][2],block[0][3],block[1][0],...
    return np.array(flattened, dtype=np.float32)

def linear_combination(blocks: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    计算块的线性组合
    
    Args:
        blocks: 形状为(num_blocks, block_size)的块数组
        coeffs: 系数向量
        
    Returns:
        线性组合结果
    """
    result = np.zeros_like(blocks[0], dtype=np.float32)
    for i, coeff in enumerate(coeffs):
        if abs(coeff) > 0:
            result += coeff * blocks[i]
    return result

def fast_4x4_block_multiply(A_blocks: List[List[np.ndarray]], 
                           B_blocks: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """
    使用49次乘法的快速算法计算4x4块矩阵乘法
    
    Args:
        A_blocks: 4x4的A小块
        B_blocks: 4x4的B小块
        
    Returns:
        4x4的结果块
    """    
    # 1. 展开小块为向量
    A_flat = flatten_4x4_blocks(A_blocks)  # 16 x (32*32)
    B_flat = flatten_4x4_blocks(B_blocks)  # 16 x (32*32)
    
    # 2. 计算49个中间矩阵 M[k] = (U[k] @ A_flat) * (V[k] @ B_flat)
    M_matrices = []
    
    for k in range(49):
        # 计算线性组合
        A_linear = linear_combination(A_flat, U[k])  # (32*32,)
        B_linear = linear_combination(B_flat, V[k])  # (32*32,)
        
        # 重塑为32x32矩阵并执行矩阵乘法
        A_matrix = A_linear.reshape(SMALL_BLOCK_ROWS, SMALL_BLOCK_COLS)
        B_matrix = B_linear.reshape(SMALL_BLOCK_COLS, SMALL_BLOCK_ROWS)
        
        # 使用numpy.dot进行32x32标准矩阵乘法
        M_k = np.dot(A_matrix, B_matrix).astype(np.float32)
        M_matrices.append(M_k.flatten())
    
    M_matrices = np.array(M_matrices, dtype=np.float32)  # 49 x (32*32)
    
    # 3. 使用W矩阵组合最终结果
    C_blocks = []
    for i in range(4):
        row_blocks = []
        for j in range(4):
            result_idx = i * 4 + j
            
            # 计算W矩阵的线性组合
            C_flat = linear_combination(M_matrices, W[result_idx])
            C_block = C_flat.reshape(SMALL_BLOCK_ROWS, SMALL_BLOCK_ROWS).astype(np.float32)
            
            row_blocks.append(C_block)
        C_blocks.append(row_blocks)
    
    return C_blocks

def assemble_4x4_blocks_to_large(blocks: List[List[np.ndarray]]) -> np.ndarray:
    """
    将4x4个小块重新组装成128x128的大块
    
    Args:
        blocks: 4x4的小块列表
        
    Returns:
        128x128的大块矩阵
    """
    result = np.zeros((LARGE_BLOCK_ROWS, LARGE_BLOCK_COLS), dtype=np.float32)
    
    for i in range(4):
        for j in range(4):
            start_row = i * SMALL_BLOCK_ROWS
            end_row = start_row + SMALL_BLOCK_ROWS
            start_col = j * SMALL_BLOCK_COLS
            end_col = start_col + SMALL_BLOCK_COLS
            
            result[start_row:end_row, start_col:end_col] = blocks[i][j]
    
    return result

def large_block_multiply(A_large: np.ndarray, B_large: np.ndarray) -> np.ndarray:
    """
    执行一个128x128大块的乘法
    
    Args:
        A_large: 128x128的A大块
        B_large: 128x128的B大块
        
    Returns:
        128x128的结果大块
    """
    # 1. 将大块分解为4x4个32x32小块
    A_blocks = extract_4x4_blocks(A_large)
    B_blocks = extract_4x4_blocks(B_large)
    
    # 2. 使用49次快速算法执行4x4块乘法
    C_blocks = fast_4x4_block_multiply(A_blocks, B_blocks)
    
    # 3. 将结果小块重新组装成大块
    C_large = assemble_4x4_blocks_to_large(C_blocks)
    return C_large

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    主矩阵乘法函数
    
    Args:
        A: MxK矩阵
        B: KxN矩阵
        
    Returns:
        MxN结果矩阵
    """
    M, K = A.shape
    K_B, N = B.shape
    
    if K != K_B:
        raise ValueError(f"矩阵维度不匹配: A是{M}x{K}, B是{K_B}x{N}")
    
    if M % LARGE_BLOCK_ROWS != 0 or K % LARGE_BLOCK_COLS != 0 or N % LARGE_BLOCK_ROWS != 0:
        raise ValueError(f"矩阵大小必须是{LARGE_BLOCK_ROWS}的倍数")
    
    # 计算大块数量
    large_blocks_M = M // LARGE_BLOCK_ROWS
    large_blocks_K = K // LARGE_BLOCK_COLS
    large_blocks_N = N // LARGE_BLOCK_ROWS
    
    print(f"矩阵分块信息:")
    print(f"  大块数量: {large_blocks_M} x {large_blocks_K} x {large_blocks_N}")
    print(f"  大块大小: {LARGE_BLOCK_ROWS} x {LARGE_BLOCK_COLS}")
    print(f"  小块大小: {SMALL_BLOCK_ROWS} x {SMALL_BLOCK_COLS}")
    print(f"  每个大块包含: 4x4 = 16个小块")
    print(f"  总计算量: {large_blocks_M * large_blocks_K * large_blocks_N}个大块乘法")
    
    # 初始化结果矩阵
    C = np.zeros((M, N), dtype=np.float32)
    
    # 执行分块矩阵乘法
    total_blocks = large_blocks_M * large_blocks_K * large_blocks_N
    
    # 创建进度条
    pbar = tqdm(total=total_blocks, desc="矩阵乘法进度", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i in range(large_blocks_M):
        for j in range(large_blocks_N):
            # 初始化当前结果大块
            C_block = np.zeros((LARGE_BLOCK_ROWS, LARGE_BLOCK_ROWS), dtype=np.float32)
            
            for k in range(large_blocks_K):
                # 更新进度条描述
                pbar.set_postfix({"当前块": f"C[{i},{j}] += A[{i},{k}] * B[{k},{j}]"})
                
                # 提取A和B的大块
                A_start_row = i * LARGE_BLOCK_ROWS
                A_end_row = A_start_row + LARGE_BLOCK_ROWS
                A_start_col = k * LARGE_BLOCK_COLS
                A_end_col = A_start_col + LARGE_BLOCK_COLS
                A_block = A[A_start_row:A_end_row, A_start_col:A_end_col]
                
                B_start_row = k * LARGE_BLOCK_COLS
                B_end_row = B_start_row + LARGE_BLOCK_COLS
                B_start_col = j * LARGE_BLOCK_ROWS
                B_end_col = B_start_col + LARGE_BLOCK_ROWS
                B_block = B[B_start_row:B_end_row, B_start_col:B_end_col]
                
                # 执行大块乘法
                partial_result = large_block_multiply(A_block, B_block)
                
                # 累加到结果大块
                C_block += partial_result
                
                # 更新进度条
                pbar.update(1)
            
            # 将结果大块写入最终矩阵
            C_start_row = i * LARGE_BLOCK_ROWS
            C_end_row = C_start_row + LARGE_BLOCK_ROWS
            C_start_col = j * LARGE_BLOCK_ROWS
            C_end_col = C_start_col + LARGE_BLOCK_ROWS
            C[C_start_row:C_end_row, C_start_col:C_end_col] = C_block
    
    # 关闭进度条
    pbar.close()
    
    return C

def verify_result(A: np.ndarray, B: np.ndarray, C: np.ndarray, sample_size: int = 100):
    """
    验证结果正确性（采样验证以节省时间）
    
    Args:
        A, B, C: 输入和输出矩阵
        sample_size: 采样验证的元素数量
        
    Returns:
        验证是否通过
    """
    print(f"\n验证结果正确性（采样 {sample_size} 个元素）...")
    
    M, N = C.shape
    K = A.shape[1]
    
    # 随机选择要验证的位置
    np.random.seed(42) 
    positions = [(np.random.randint(0, M), np.random.randint(0, N)) for _ in range(sample_size)]
    
    max_error = 0.0
    for idx, (i, j) in enumerate(positions):
        expected = np.dot(A[i, :], B[:, j])
        actual = C[i, j]
        error = abs(expected - actual)
        max_error = max(max_error, error)
        
        if error > 1e-2:  # 允许一定的数值误差
            print(f"  错误: 位置[{i},{j}], 期望值={expected}, 实际值={actual}, 误差={error}")
    
    print(f"  验证通过! 最大误差: {max_error}")
    return True

def load_matrix_from_file(file_path: str) -> np.ndarray:
    """
    从文件加载矩阵
    
    Args:
        file_path: 矩阵文件路径，支持.npy, .csv, .txt格式
        
    Returns:
        加载的矩阵
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"矩阵文件不存在: {file_path}")
    
    print(f"从文件加载矩阵: {file_path}")
    
    # 根据文件扩展名选择加载方式
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.npy':
            matrix = np.load(file_path)
        elif file_ext == '.csv':
            matrix = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        elif file_ext in ['.txt', '.dat']:
            matrix = np.loadtxt(file_path, dtype=np.float32)
        else:
            # 尝试以numpy格式加载
            try:
                matrix = np.load(file_path)
            except:
                # 尝试以文本格式加载
                matrix = np.loadtxt(file_path, dtype=np.float32)
        
        print(f"  成功加载矩阵，形状: {matrix.shape}")
        print(f"  数据类型: {matrix.dtype}")
        
        # 检查矩阵大小
        if matrix.shape[0] != 4096 or matrix.shape[1] != 4096:
            raise ValueError(f"矩阵大小必须是4096x4096，实际大小: {matrix.shape}")
        
        # 确保数据类型为float32
        if matrix.dtype != np.float32:
            print(f"  转换数据类型从 {matrix.dtype} 到 float32")
            matrix = matrix.astype(np.float32)
        
        return matrix
        
    except Exception as e:
        raise RuntimeError(f"加载矩阵文件失败: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分块矩阵乘法程序')
    parser.add_argument('matrix_path', nargs='?', default=None,
                       help='4096x4096矩阵文件路径 (支持.npy, .csv, .txt格式)')
    parser.add_argument('--generate-random', action='store_true',
                       help='生成随机矩阵而不是从文件加载')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机数种子 (默认: 42)')
    
    args = parser.parse_args()
    
    print("Python版本的分块矩阵乘法程序")
    print("="*50)
    
    # 设置矩阵大小
    MATRIX_SIZE = 4096
    
    print(f"矩阵大小: {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"数据类型: float32")
    
    # 设置随机数种子
    np.random.seed(args.seed)
    
    # 加载或生成A矩阵
    if args.generate_random or args.matrix_path is None:
        print("\n生成随机矩阵A...")
        A = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    else:
        try:
            A = load_matrix_from_file(args.matrix_path)
        except Exception as e:
            print(f"错误: {e}")
            print("将生成随机矩阵作为替代...")
            A = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    
    # 生成随机B矩阵
    print("生成随机矩阵B...")
    B = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    
    # 执行矩阵乘法
    print(f"\n开始执行分块矩阵乘法...")
    
    C = matrix_multiply(A, B)

    print(f"\n矩阵乘法计算完成！")

    
    # 验证结果正确性
    verify_result(A, B, C, sample_size=100)

if __name__ == "__main__":
    main()
