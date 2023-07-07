import numpy as np
import math

def radius(data, MinPts):
    m, n = np.shape(data)#m:样本个数；n:特征的维度
    Mymax_data = np.max(data, 0)#找到最大与最小的样本
    Mymin_data = np.min(data, 0)
    r = (((np.prod(Mymax_data - Mymin_data) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n))
    return r