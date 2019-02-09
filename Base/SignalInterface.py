import numpy as np
from deprecated import deprecated
from scipy.io import loadmat
from scipy.signal import fftconvolve
from scipy.constants import c
import os
import visdom

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Signal(object):

    def __init__(self, **kwargs):
        '''

        :param kwargs: 包含单信道信号的各项参数，包括：
            1. 符号速率 【Hz】
            2. 调制格式
            3. 符号长度
            4. 符号功率  【dbm】
            5. sps： 发端dsp的采样率
            6. sps_in_fiber: 光纤中采样率
            7. lam： 信号波长，单位为国际单位m

        所有单位全部使用国际标准单位

        '''
        if isinstance(self, WdmSignal):
            return

        self.symbol_rate = kwargs['symbol_rate']
        self.mf = kwargs['mf']
        self.symbol_length = kwargs['symbol_length']

        self.signal_power = None

        self.sps_in_fiber = kwargs['sps_in_fiber']
        self.sps = kwargs['sps']
        self.lamb = None
        self.data_sample = None
        self.is_dp = True
        self.data_sample_in_fiber = None

    def tonumpy(self):
        '''

        :return: reference to self.data_sample_in_fiber
        if function not return a new ndarrya
        change it outside will change the data_sample_in_fiber property of the object
        '''
        return self.data_sample_in_fiber

    def __getitem__(self, index):
        '''

        :param index: Slice Object
        :return: ndarray of the signal, change this ndarray will not change the object

        '''
        assert self.data_sample_in_fiber is not None

        return self.data_sample_in_fiber[index]

    def __setitem__(self, index, value):
        '''

        :param index: SLICE Object
        :param value: the value that to be set
        :return:
        '''
        if self.data_sample_in_fiber is None:
            self.data_sample_in_fiber = np.atleast_2d(value)
        else:

           self.data_sample_in_fiber[index] = value


    @staticmethod
    def lamb2freq(lam):
        '''

        :param lam: wavelength [m]
        :return: frequence [Hz]
        '''
        return c/lam

    @staticmethod
    def freq2lamb(freq):
        '''

        :param freq: frequence [Hz]
        :return: lambda:[m]
        '''
        return c/freq


    def measure_power_in_fiber(self):
        sample_in_fiber = np.atleast_2d(self.data_sample_in_fiber)
        signal_power = np.sum(np.mean(np.abs(sample_in_fiber) ** 2, axis=1))
        return signal_power

    @property
    def fs(self):
        if self.sps is None:
            return 0
        return self.symbol_rate * self.sps

    @property
    def fs_in_fiber(self):
        return self.symbol_rate * self.sps_in_fiber

    @deprecated(version='0.1',reason='please use measure_power_in_fiber')
    def avg_power(self):
        x_power = np.mean(np.abs(self.data_sample_in_fiber[0, :]) ** 2)
        y_power = np.mean(np.abs(self.data_sample_in_fiber[1, :]) ** 2)
        return x_power + y_power



    def _normalize_power(self):
        '''
            in place operation
        :return:
        '''
        self.data_sample_in_fiber = self.normalize_power()

    def normalize_power(self):
        '''

        new ndarray will be returned , the signal object itself is not changed
        :param signal:
        :return: ndarray and signal object will not be changed

        '''
        if self.data_sample_in_fiber.shape[0] == 2:
            x_normal = self.data_sample_in_fiber[0,:] / np.sqrt(np.mean(np.abs(self.data_sample_in_fiber[0, :]) ** 2))
            y_normal = self.data_sample_in_fiber[1, :] / np.sqrt(np.mean(np.abs(self.data_sample_in_fiber[1, :]) ** 2))
            return x_normal, y_normal
        else:
            normal_signal_sample = np.atleast_2d(self.data_sample_in_fiber)
            normal_signal_sample = normal_signal_sample[0, :] / np.sqrt(
                np.mean(np.abs(normal_signal_sample[0, :]) ** 2))

            return np.atleast_2d(normal_signal_sample)

    def from_numpy(self, **kwargs):
        if kwargs['is_wdm']:
            if not isinstance(kwargs['mf'], list):
                mf = [kwargs['mf']]
            else:
                mf = kwargs['mf']
            fs = kwargs['fs']
            wdm_sample = kwargs['wdm_sample']
            if kwargs['is_dp']:
                sample_x = wdm_sample[0, :]
                sample_y = wdm_sample[1, :]
            else:
                sample_x = wdm_sample[0, :]
                sample_y = None

            center_freq = kwargs['center_freq']
            spacing = kwargs['spacing']
            symbol_rate = kwargs['symbol_rate']
            absolute_frequence = kwargs['absolute_frequence']
            ch_number = kwargs['ch_number']

            return WdmSignal(
                **dict(sample_x=sample_x, sample_y=sample_y, mf=mf, spacing=spacing, center_freq=center_freq,
                       symbol_rate=symbol_rate, absolute_frequence=absolute_frequence, ch_number=ch_number, fs=fs))

        else:
            symbol_rate = kwargs['symbol_rate']
            mf = kwargs['mf']
            signal_power = kwargs['signal_power']
            symbol_length = kwargs['symbol_length']

            if 'sample_x' in kwargs:
                sps = kwargs['sps']

            else:
                sps = None
            sps_in_fiber = kwargs['sps_in_fiber']

            qamsignal = QamSignal(symbol_rate=symbol_rate, mf=mf, signal_power=signal_power,
                                  symbol_length=symbol_length, sps=sps, sps_in_fiber=sps_in_fiber, is_from_numpy=True)

            symbol_x = kwargs['symbol_x']
            if kwargs['is_dp']:
                symbol_y = kwargs['symbol_y']
                qamsignal.symbol = np.array([symbol_x, symbol_y])
            else:
                qamsignal.symbol = np.atleast_2d(np.array(symbol_x))

            if sps:
                sample_x = kwargs['sample_x']
                if kwargs['is_dp']:

                    sample_y = kwargs['sample_y']

                    sample = np.array([sample_x, sample_y])
                    qamsignal.data_sample = sample
                else:
                    qamsignal.data_sample = np.atleast_2d(np.array(sample_x))
            sample_x_in_fiber = kwargs['sample_x_in_fiber']
            if kwargs['is_dp']:
                sample_y_in_fiber = kwargs['sample_y_in_fiber']

                qamsignal.data_sample_in_fiber = np.array([sample_x_in_fiber, sample_y_in_fiber])
            else:
                qamsignal.data_sample_in_fiber = np.atleast_2d(np.array(sample_x_in_fiber))

            return qamsignal

    def upsample(self, symbol_x, sps):
        symbol_x.shape = -1, 1
        symbol_x = np.tile(symbol_x, (1, sps))
        symbol_x[:, 1:] = 0
        symbol_x.shape = 1, -1
        return symbol_x

    def __str__(self):

        string = f'\n\tSymbol rate:{self.symbol_rate/1e9}[Ghz]\n\tfs_in_fiber:{self.fs_in_fiber/1e9}[Ghz]\n\tsignal_power_in_fiber:{self.signal_power} [W]\n\tsignal WaveLength:{self.lamb*1e9}[nm] '

        return string

    def __repr__(self):

        return self.__str__()

class WdmSignal(object):

    def __init__(self, sample_x, sample_y=None, **kwargs):
        '''

        :param sample_x:
        :param sample_y:
        :param kwargs:
        '''
        if kwargs['is_dp']:
            assert sample_y is not None
            self.data_sample = np.array([sample_x, sample_y])
        else:
            self.data_sample = np.array([sample_x])

        self.center_freq = kwargs['center_freq']
        self.spacing = kwargs['spacing']
        self.mf = kwargs['mf']
        self.symbol_rate = kwargs['symbol_rate']

        self.absolute_freq = kwargs['absolute_frequence']  # 指的是以0 为中心
        self.ch_number = kwargs['ch_number']
        self.fs = kwargs['fs']


class QamSignal(Signal):

    def __init__(self, symbol_rate, mf, symbol_length, sps, sps_in_fiber, is_dp=True,
                 is_from_numpy=False):
        '''

        :param symbol_rate: [hz]   符号速率
        :param mf: 调制格式
        :param signal_power: [dbm] 信号功率
        :param symbol_length:      符号长度
        :param sps: 发端dsp的过采样率
        :param sps_infiber: 在光纤中传输时的过采样率
        '''

        super().__init__(
            **dict(symbol_length=symbol_length, symbol_rate=symbol_rate, mf=mf,  sps=sps,
                   sps_in_fiber=sps_in_fiber, is_dp=is_dp)
        )

        if self.mf == 'qpsk':
            order = 4
        else:
            order = self.mf.split('-')[0]
            order = int(order)

        if self.is_dp:

            self.message = np.random.randint(low=0, high=order, size=(2, self.symbol_length))
        else:
            self.message = np.random.randint(low=0, high=order, size=(1, self.symbol_length))

        self.symbol = None

        if not is_from_numpy:
            self.init(order)

    def init(self, order):

        qam_data = (f'{base_path}/' + 'qamdata/' + str(order) + 'qam.mat')
        qam_data = loadmat(qam_data)['x']

        if self.is_dp:
            symbol = np.zeros((2, self.symbol_length), dtype=np.complex)
            for index, msg in enumerate(self.message[0, :]):
                symbol[0, index] = qam_data[0, msg]
                symbol[1, index] = qam_data[0, self.message[1, index]]
            print('------symbol_map completed------')
            self.symbol = symbol
        else:
            symbol = np.zeros((1, self.symbol_length), dtype=np.complex)
            for index, msg in enumerate(self.message[0, :]):
                symbol[0, index] = qam_data[0, msg]

            print('------symbol_map completed------')
            self.symbol = symbol


def main():
    # symbol_rate, mf, signal_power, symbol_length, sps, sps_infiber
    symbol_rate = 35e9
    mf = '16-qam'
    signal_power = 0
    symbol_length = 5
    sps = 2
    sps_infiber = 4

    parameter = dict(symbol_rate=symbol_rate, mf=mf,  symbol_length=symbol_length, sps=sps,
                     sps_in_fiber=sps_infiber)

    signal = QamSignal(**parameter)
    signal[:] = np.array([1,2,3,4,5])
    signal[0,:] = np.array([4,5,6,7,8])
    # signal._normalize_power()
    print(signal[0,:])
    print(signal.measure_power_in_fiber())
    print("heiehi")


if __name__ == '__main__':
    main()
