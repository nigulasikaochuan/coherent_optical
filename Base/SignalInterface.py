import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve
import os
import visdom

base_path = os.path.dirname(os.path.abspath(__file__))


class Signal(object):

    # def __init__(self):
    #     pass

    def plot_spectrum(self, vis: visdom.Visdom):
        print('dual-pol')
        x_sample = self.data_sample_infiber[0, :]
        y_sample = self.data_sample_infiber[1, :]
        fs = self.fs

        import matplotlib.pyplot as plt
        figure = plt.figure()
        ax = figure.add_subplot(121)
        ax.psd(x_sample, NFFT=len(x_sample), Fs=fs, sides='twosided')
        plt.title('x-pol')
        ax = figure.add_subplot(122)
        ax.psd(x_sample, NFFT=len(y_sample), Fs=fs, sides='twosided')
        plt.title('y-pol')
        vis.matplot(plt)

    @property
    def fs(self):
        return self.symbol * self.sps_in_fiber

    def avg_power(self):
        x_power = np.mean(np.abs(self.data_sample_infiber[0, :]) ** 2)
        y_power = np.mean(np.abs(self.data_sample_infiber[1, :]) ** 2)
        return x_power + y_power

    def normalize_power(self):
        '''

        :param signal:
        :return: ndarray and signal object will not be changed
        '''
        if self.data_sample.shape[0] == 2:
            x_normal = self.data_sample[0, :] / np.mean(np.abs(self.data_sample[0, :]) ** 2)
            y_normal = self.data_sample[1, :] / np.mean(np.abs(self.data_sample[1, :]) ** 2)
            return x_normal, y_normal
        else:
            normal_signal_sample = np.atleast_2d(self.data_sample)
            normal_signal_sample = normal_signal_sample[0, :] / np.mean(np.abs(normal_signal_sample[0, :]) ** 2)

            return normal_signal_sample

    def from_numpy_array(self,sample_x,sample_y,mf=None,symbol_rate=None,sps=None,wdm_signal=False,center_freq = 193.1e12,spacing=50e9
                         ,ch_number = 1,absolute_frequence = []):



        if wdm_signal:
            assert ch_number > 1
            assert len(absolute_frequence) > 0
            if symbol_rate is None:
                symbol_rate = 35e9
                print("Warning: Symbol rate is set to 35GBAUD for each channel")

            if isinstance(symbol_rate,list):
                print('each channel have different rate')

            else:
                print('each channel have same rate')

            if isinstance(mf,list):
                print('each channel have different mf')
            else:
                print('each channel have the same mf')

            config_parameter = dict(center_freq = center_freq,spacing = spacing,mf = mf, symbol_rate=symbol_rate,ch_number=ch_number,
                                    absolute_frequence = absolute_frequence)

            return WdmSignal(sample_x,sample_y,**config_parameter)
        else:

            if mf in ['qpsk','16qam','32qam','64qam','8qam']:
                pass


class WdmSignal(Signal):

    def __init__(self,sample_x,sample_y,**kwargs):
        super().__init__()
        self.sample_x = sample_x
        self.sample_y = sample_y

        self.center_freq = kwargs['center_freq']
        self.spacing = kwargs['spacing']
        self.mf = kwargs['mf']
        self.symbol_rate = kwargs['symbol_rate']
        self.absolute_freq = kwargs['absolute_frequence']
        self.ch_number = kwargs['ch_number']




class QamSignal(Signal):

    def __init__(self, symbol_rate, mf, signal_power, symbol_length, sps):

        self.symbol_rate = symbol_rate
        self.mf = mf
        if self.mf == 'qpsk':
            order = 4
        else:
            order = self.mf.split('-')[0]
            order = int(order)
        self.symbol_length = symbol_length

        self.signal_power = signal_power
        self.message = np.random.randint(low=0, high=order, size=(2, self.symbol_length))

        self.symbol = None
        self.data_sample_infiber = None
        self.rrc_filter_tap = None

        self.sps_in_fiber = sps
        self.lamb = None
        self.init(order)



    def init(self, order):

        qam_data = (f'{base_path}/' + str(order) + 'qam.mat')
        qam_data = loadmat(qam_data)['x']

        symbol = np.zeros((2, self.symbol_length), dtype=np.complex)
        for index, msg in enumerate(self.message[0, :]):
            symbol[0, index] = qam_data[msg]
            symbol[1, index] = qam_data[self.message[1, index]]
        print('------symbol_map completed------')

        ### 插零
        symbol_x = self.symbol[0, :]
        symbol_y = self.symbol[1, :]

        symbol_x = self.upsample(symbol_x)
        symbol_y = self.upsample(symbol_y)
        self.rrc_filter_tap = self.rrc_filter(0.02, self.sps_in_fiber, 1024)

        symbol_x = fftconvolve(symbol_x[0], self.rrc_filter_tap[0])
        symbol_y = fftconvolve(symbol_y[0], self.rrc_filter_tap[0])

        self.data_sample_infiber = np.array([symbol_x, symbol_y])

    def upsample(self, symbol_x):
        symbol_x.shape = -1, 1
        symbol_x = np.tile(symbol_x, (1, self.sps_in_fiber))
        symbol_x[:, 1:] = 0
        symbol_x.shape = 1, -1
        return symbol_x

    def rrc_filter(self, roll_off, sps, span):
        pass


