'''
    仿真时候，采用离散的样本点，假设符号率为R GHZ,每个符号用sps个样点来表示。所以两个样点之间的时间间隔为
        1/R/sps/1e9 (s)
    采样频率为：
        fs = R*sps GHZ
    总的时间跨度为:
        range(0,(sps)*nsymbol*fs/N,fs/N)

'''

from Base.SignalInterface import QamSignal
from oInstrument.OptiLine import LaserSource
import numpy as np
if __name__ == '__main__':
    symbol_rate = 35e9
    mf = '16-qam'
    signal_power = 0
    symbol_length = 2 ** 16
    sps = 2
    sps_infiber = 4

    parameter = dict(symbol_rate=symbol_rate, mf=mf, symbol_length=symbol_length, sps=sps,
                     sps_in_fiber=sps_infiber)

    signal = QamSignal(**parameter)
    signal.data_sample_in_fiber = np.array([[1, 2, 3, 4, 5]])

    laser = LaserSource(0, 0.001, True, 193.1e12)
    laser(signal)


