# 化成预筛选数据集

该文件夹下为数据集相关文件，具体文件说明如下
* ___data_set.pt___
模型所用数据集文件。
 数据类型：python.dict
|    key     |      value type     |   size (n_samples, features)       | note            |
|------------|-------------------|-----------------|----------------------|
  | 'train df' |    pandas.DataFrame | (28160, 162)                       | train data      |
   | 'val df'   |    pandas.DataFrame | (9384, 162)                        | validation data |
   | 'test df'  |    pandas.DataFrame | (9384, 162)                        | test data       |

* ___data_set.png___
数据集展示图。该数据集中每条线为一个电池化成充电电压数据。整个数据集分成三部分：训练集，验证集，测试集。原始数据经过标准化，裁剪的得到数据集。
**train df**:训练数据集。
**val df**:验证数据集。
**test df**:测试数据集。