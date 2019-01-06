from Base.SignalInterface import Signal
from scipy.fftpack import  fftfreq
from scipy.fftpack import fft
from scipy.fftpack import ifft
class AnotherMethodException(Exception):
    pass


class Filter:

    def __init__(self, filter_type):
        self.filter_type = filter_type


class LowPassFilter(Filter):

    def __init__(self, filter_type, bw):
        super(LowPassFilter, self).__init__(filter_type)
        self.bw = bw

    def _ideal_lowpass(self,signal):
        '''

        :param signal: in place filter
        :return:
        '''
        sample_x,sample_y = self.ideal_lowpass(signal)
        signal.data_sample[0,:] = sample_x
        signal.data_sample[1,:] = sample_y

    def ideal_lowpass(self, signal: Signal):
        '''

        :param signal: the sampled sample will be returned, original sample of signal will not be changed
        :return:
        '''
        fs = signal.fs
        sample_x = signal.data_sample[0, :]
        sample_y = signal.data_sample[1, :]
        freq = fftfreq(signal.data_sample.shape[1],1/fs)

        sample_x_fourier_transform = fft(sample_x)
        sample_y_fourier_transform = fft(sample_y)

        # 超过滤波器带宽的频率点直接设置为0
        sample_x_fourier_transform[abs(freq) > self.bw] = 0
        sample_y_fourier_transform[abs(freq)>self.bw] = 0

        sample_x = ifft(sample_x_fourier_transform)
        sample_y = ifft(sample_y_fourier_transform)
        return sample_x,sample_y


    def design(self):
        if self.filter_type == 'ideal':

            raise AnotherMethodException(
                "using fft method, the spectrum points is manipultated directly,please use ideal_filter method")

        if self.filter_type == 'Bessel':
            pass

        if self.filter_type == 'Gaussian':
            pass

