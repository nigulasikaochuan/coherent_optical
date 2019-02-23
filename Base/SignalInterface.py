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
            1. symbol_rate 符号速率 【Hz】
            2. mf 调制格式
            3. symbol length符号长度
            4. signal_power 信号功率  【dbm】
            5. sps： 发端dsp的采样率
            6. sps_in_fiber: 光纤中采样率
            7. lam： 信号波长，单位为国际单位 m

        所有单位全部使用国际标准单位

        '''
        if isinstance(self, WdmSignal):
            return

        self.symbol_rate = kwargs['symbol_rate']
        self.mf = kwargs['mf']
        self.symbol_length = kwargs['symbol_length']

        self.signal_power = None
        # self.center_frequence = None # will be set in laser
        self.sps_in_fiber = kwargs['sps_in_fiber']
        self.sps = kwargs['sps']
        self.lamb = None

        self.data_sample = None
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
        :return: ndarray of the signal, change this ndarray will change the object

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
        '''

        :return: signal power in the fiber, in unit [W]
        '''
        sample_in_fiber = np.atleast_2d(self.data_sample_in_fiber)
        signal_power = 0
        for i in range(sample_in_fiber.shape[0]):
            signal_power += np.mean(np.abs(sample_in_fiber[i,:])**2)

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



    def inplace_normalize_power(self):
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
        temp = np.zeros_like(self.data_sample_in_fiber)
        for i in range(self.data_sample_in_fiber.shape[0]):
            temp[i,:] = self.data_sample_in_fiber[i,:]/np.sqrt(np.mean(np.abs(self.data_sample_in_fiber[i,:])**2))


        return temp


    def upsample(self, symbol_x, sps):
        '''

        :param symbol_x: 1d array
        :param sps: sample per symbol
        :return: 2-d array after inserting zeroes between symbols
        '''
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

class ReceiveSignal(object):

    def __init__(self,symbol):
        self.symbol = symbol

class SignalFromNumpyArray(Signal):

    def __init__(self,array_sample,symbol_rate,mf,symbol_length,sps_in_fiber,sps):
        '''

        :param array_sample: nd arrary
        :param symbol_rate: GHZ
        :param mf:
        :param symbol_length:
        :param sps_in_fiber:
        :param sps:
        '''
        super().__init__(**dict(symbol_rate = symbol_rate,mf=mf,symbol_length = symbol_length,sps_in_fiber = sps_in_fiber,sps=sps))
        self.data_sample_in_fiber = array_sample

class WdmSignal(object):
    '''
        WdmSignal Class
    '''
    def __init__(self,signals,grid_size,center_frequence):
        '''

        :param signals: list of signal
        :param grid_size: [GHz]
        '''
        self.signals = signals
        self.grid_size = grid_size
        self.center_frequence = center_frequence
    @property
    def fs(self):
        return self.signals[0].fs_in_fiber

    @property
    def mfs(self):
        mfs = []
        for sig in self.signals:
            mfs.append(sig.mf)

    @property
    def absolute_frequences(self):
        '''

        :return:
        '''
        releative = self.relative_frequence
        releative = np.array(releative)
        return self.center_frequence + releative

    @property
    def relative_frequence(self):
        '''

        :return: frequences centered in zero frequence , a List
        '''
        pos_frequence = []

        if divmod(len(self.signals),2)[1] == 0:
            each_number = int(len(self.signals)/2)
            pos_frequence_first = self.grid_size/2
            pos_frequence.append(pos_frequence_first)
            for i in range(0,each_number-1):
                pos_frequence.append(pos_frequence_first + (i+1)*self.grid_size)

            neg_frequence = pos_frequence[::-1]

            neg_frequence = np.array(neg_frequence)*-1
            neg_frequence = neg_frequence.tolist()

            return neg_frequence+pos_frequence

        else:
            each_number = len(self.signals)-1
            each_number = int(each_number/2)

            for i in range(each_number):
                pos_frequence.append((i+1)*self.grid_size)
            neg_frequence = pos_frequence[::-1]
            neg_frequence = np.array(neg_frequence)*-1
            neg_frequence = neg_frequence.tolist()
            neg_frequence.append(0)
            return neg_frequence+pos_frequence




class QamSignal(Signal):

    def __init__(self, symbol_rate, mf, symbol_length, sps, sps_in_fiber,
                 is_from_numpy=False,is_dp=True):
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
                   sps_in_fiber=sps_in_fiber)
        )

        if self.mf == 'qpsk':
            order = 4
        else:
            order = self.mf.split('-')[0]
            order = int(order)

        if is_dp:

            self.message = np.random.randint(low=0, high=order, size=(2, self.symbol_length))
        else:
            self.message = np.random.randint(low=0, high=order, size=(1, self.symbol_length))
        self.is_dp = is_dp
        self.symbol = None

        if not is_from_numpy:
            self.init(order)

    def init(self, order):

        qam_data = (f'{base_path}/' + 'qamdata/' + str(order) + 'qam.mat')
        qam_data = loadmat(qam_data)['x']

        symbol = np.zeros(self.message.shape,dtype=np.complex)

        for i in range(symbol.shape[0]):
            for msg in np.unique(self.message[i,:]):
                symbol[i,self.message[i,:] == msg] = qam_data[0,msg]

        self.symbol = symbol

        self.data_sample = np.zeros((symbol.shape[0],symbol.shape[1]*self.sps),dtype=np.complex)
        self.data_sample_in_fiber = np.zeros((symbol.shape[0],symbol.shape[1]*self.sps_in_fiber),dtype=np.complex)

def main():
    # symbol_rate, mf, signal_power, symbol_length, sps, sps_infiber
    symbol_rate = 35e9
    mf = '16-qam'
    signal_power = 0
    symbol_length = 5
    sps = 2
    sps_infiber = 4

    parameter = dict(symbol_rate=symbol_rate, mf=mf,  symbol_length=symbol_length, sps=sps,
                     sps_in_fiber=sps_infiber,is_dp = False)

    signal = QamSignal(**parameter)
    signal[:] = np.array([1,2,3,4,5])
    signal[0,:] = np.array([4,5,6,7,8])
    # signal._normalize_power()
    print(signal[0,:])
    print(signal.measure_power_in_fiber())
    print("heiehi")


if __name__ == '__main__':
    main()
