from typing import List

import numpy as np
class Signal():
    pass

def mux_signal(signals:List):
    absolute_frequences = []
    for signal in signals:
        absolute_frequences.append(signal.absolute_frequence)

    # 从左到右开始复用
    # 每个wdm信道的采样频率必须保持一致
    # 每个wdm信道的样本点个数也要保持一致,截掉长的采样序列的尾部

    sample_length = []
    for signal in signals:
        sample_length.append(signals.data_sample.shape[1])

    min_length = min(sample_length)

    for signal in signals:
        number_to_del = signal.data_sample.shape[1] - min_length
        if number_to_del == 0:
            continue
        else:
            signal.data_sample_infiber  = signal.data_sample_infiber[:,1:-1+number_to_del]
    fs = signals[0].sps*signals[0].symbol_rate
    t = np.arange(0,min_length-1,1)*fs/min_length

    wdm_datasample = 0
    for index,signal in enumerate(signals):
        wdm_datasample+=signal.data_sample_infiber*np.exp(1j*2*np.pi*absolute_frequences[index]*t)

    wdm_signal = Signal()
    wdm_signal.symbol_rate = [signal.symbol_rate for signal in signals]
    wdm_signal.mf = [signal.mf for signal in signals]
    return wdm_signal
