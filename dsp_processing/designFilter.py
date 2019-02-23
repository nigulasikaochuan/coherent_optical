from Base.SignalInterface import Signal
from scipy.fftpack import fftfreq
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import iirfilter
from scipy.signal import group_delay
from scipy.signal import fftconvolve
import numpy as np

from eInstrument.ElectricInstrument import PulseShaping


class AnotherMethodException(Exception):
    pass


class Filter:

    def __init__(self, filter_type):
        self.filter_type = filter_type


class LowPassFilter(Filter):

    def __init__(self, filter_type, bw, order=None, fs=0):
        super(LowPassFilter, self).__init__(filter_type)
        self.bw = bw
        self.order = order
        assert fs != 0
        self.fs = fs

    def inplace_ideal_lowpass(self, signal):
        '''

        :param signal: in place filter
        :return:
        '''
        sample_x, sample_y = self.ideal_lowpass(signal)
        signal.data_sample[0, :] = sample_x
        signal.data_sample[1, :] = sample_y

    def ideal_lowpass(self, signal: Signal):
        '''

        :param signal: the sampled sample will be returned, original sample of signal will not be changed
        :return:
        '''
        fs_in_fiber = signal.fs_in_fiber
        sample_x = signal.data_sample_in_fiber[0, :]
        sample_y = signal.data_sample_in_fiber[1, :]
        freq = fftfreq(signal.data_sample.shape[1], 1 / fs_in_fiber)

        sample_x_fourier_transform = fft(sample_x)
        sample_y_fourier_transform = fft(sample_y)

        # 超过滤波器带宽的频率点直接设置为0
        sample_x_fourier_transform[abs(freq) > self.bw] = 0
        sample_y_fourier_transform[abs(freq) > self.bw] = 0

        sample_x = ifft(sample_x_fourier_transform)
        sample_y = ifft(sample_y_fourier_transform)
        return sample_x, sample_y

    def design(self):
        if self.filter_type == 'ideal':
            raise AnotherMethodException(
                "using fft method, the spectrum points is manipultated directly,please use ideal_filter method")

        if self.filter_type == 'Bessel':
            assert self.order is not None

            b, a = iirfilter(self.order, [self.bw], btype='lowpass', ftype=self.filter_type, fs=self.fs)

        if self.filter_type == 'Gaussian':
            pass


class MatchedFilter(object):

    def __init__(self, roll_off,span,sps):
        self.h = PulseShaping.rcosdesign(span,roll_off,1,sps)
        self.delay = int(span/2 * sps)

    def match_filter(self, signal):
        x = signal.data_sample_infiber[0, :]
        y = signal.data_sample_infiber[1, :]
        x = fftconvolve(x,self.h)
        y = fftconvolve(y,self.h)

        x = np.roll(x,-self.delay)
        y = np.roll(y,-self.delay)
        x = x[:len(x)]
        y = y[:len(y)]
        return x,y

    def inplace_match_filter(self, signal):
        x, y = self.match_filter(signal)
        signal.data_sample = np.array([x, y])

    def __call__(self, signal):
        self.inplace_match_filter(signal)


def fvtool(b, a, fs, env='heihei'):
    from scipy.signal import freqz
    w, h = freqz(b, a, whole=True, fs=fs)
    import visdom
    import plotly
    vis1 = visdom.Visdom(env=env)


