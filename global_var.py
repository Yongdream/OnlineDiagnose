# coding=utf-8
# 在别的文件使用方法:
# import global_var_model as gl
#  gl.gl_int_i += 4,可以通过访问和修改gl.gl_int_i来实现python的全局变量,或者叫静态变量访问
# gl.gl_int_i
import numpy as np
gl_int_i = 1  # 这里的gl_int_i是最常用的用于标记的全局变量
gl_str_i = 'one'
gl_str_i1 = 'one'
gl_str_i2 = 'one'
gl_str_i3 = 'one'
gl_str_i4 = 'one'
batch_size = 1
Epoch = 1
model = 'MSCRED'
learning_rate = 0.1
i = 0
dataset='MBA'
Y = np.array([])
