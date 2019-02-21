from numpy.fft import fftfreq
from scipy.fftpack import fft, ifft

from Base import SignalInterface
import numpy as np
from scipy.constants import c


class Awgn(object):

    def __init__(self, signal, request_snr, sps):
        self.sps = sps
        if isinstance(signal, SignalInterface.Signal):
            self.x_pol = signal.data_sample[0, :]
            self.y_pol = signal.data_sample[1, :]

        else:
            signal = np.atleast_2d(signal)

        self.request_snr = request_snr

    def __call__(self):

        power_x = np.mean(np.abs(self.x_pol) ** 2)
        power_y = np.mean(np.abs(self.y_pol) ** 2)

        total_power = power_x + power_y
        noise_power = total_power / self.request_snr_lin

        noise_x = np.sqrt(noise_power / 2 / 2) * (
                np.random.randn((self.x_pol).shape) + 1j * np.random.randn(self.x_pol.shape)) * self.sps

        noise_y = np.sqrt(noise_power / 2 / 2) * (
                np.random.randn((self.y_pol).shape) + 1j * np.random.randn(self.y_pol.shape)) * self.sps

        self.x_pol += noise_x
        self.y_pol += noise_y

        return self.x_pol, self.y_pol

    @property
    def request_snr_lin(self):

        return 10 ** (self.request_snr / 10)


class LinearFiber(object):
    '''
        property:
            self.alpha  [db/km]
            self.D [ps/nm/km]
            self.length [km]

            self.beta2: caculate beta2 from D,s^2/km
        method:
            __call__: the signal will
    '''

    def __init__(self, alpha, D, length, wave_length=1550):
        self.alpha = alpha
        self.D = D
        self.length = length
        self.wave_length = wave_length

    @property
    def beta2(self):
        return -self.D * (self.wave_length * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def prop(self, signal):
        after_prop = np.zeros_like(signal.data_sample)
        for pol in range(0, signal.data_sample.shape[0]):
            sample = signal.data_sample[pol, :]
            sample_fft = fft(sample)
            freq = fftfreq(signal.data_sample.shape[1], 1 / signal.fs)
            transfer_function = np.exp(
                -self.alpha * self.length / 2 + 1j * self.beta2 * (freq * 2 * np.pi) ** 2 * self.length / 2)
            after_prop[pol, :] = ifft(sample_fft * transfer_function)
        return after_prop

    def inplace_prop(self, signal):
        after_prop = self.prop(signal)
        self.data_sample = after_prop

    def __call__(self, signal):
        self.inplace_prop(signal)


import arrayfire as af


class NonlinearFiber(LinearFiber):

    def __init__(self, alpha, D, length, gamma, step_length):
        super().__init__(alpha, D, length)
        self.gamma = gamma
        self.step_length = step_length

    @property
    def step_length_eff(self):
        return (1 - np.exp(-self.alpha * self.step_length)) / self.alpha

    def prop(self, signal):
        if signal.ndim != 2:
            raise Exception("only dp signal supported at this time")
        step_number = self.length // self.step_length

        temp = np.zeros_like(signal.data_sample)
        freq = fftfreq(signal.data_sample.shape[1], 1 / signal.fs)
        freq_gpu = af.np_to_af_array(freq)

        D = 1j * self.beta2 / 2 * 2 * np.pi * freq_gpu ** 2
        N = 8 / 9 * 1j * self.gamma
        atten = -self.alpha / 2

        time_x = af.np_to_af_array(signal.data_sample[0, :])
        time_y = af.np_to_af_array(signal.data_sample[1, :])

        print('*' * 20 + "begin simulation" + '*' * 20)
        for i in range(step_number):
            fftx = af.fft(time_x)
            ffty = af.fft(time_y)

            time_x = af.ifft(fftx * D * self.step_length / 2) * af.exp(atten) * af.exp(
                N * (af.abs(time_x[1, :]) ** 2 + af.abs(time_y[0, :]) ** 2)) * self.step_length_eff
            time_y = af.ifft(ffty * D * self.step_length / 2) * af.exp(atten) * af.exp(
                N * (af.abs(time_x[1, :]) ** 2 + af.abs(time_y[0, :]) ** 2)) * self.step_length_eff

            fftx = af.fft(time_x)
            ffty = af.fft(time_y)
            time_y = af.ifft(ffty * D * self.step_length / 2)
            time_x = af.ifft(fftx * D * self.step_length / 2)

        print('*' * 20 + "end simulation" + '*' * 20)

        last_step = self.length - self.step_length * step_number
        last_step_eff = (1 - np.exp(-self.alpha * last_step)) / self.alpha
        if last_step == 0:
            temp[0, :] = time_x.to_ndarray()
            temp[1, :] = time_y.to_ndarray()
            return temp

        fftx = af.fft(time_x)
        ffty = af.fft(time_y)

        time_x = af.ifft(fftx * D * last_step / 2) * af.exp(atten) * af.exp(
            N * (af.abs(time_x[1, :]) ** 2 + af.abs(time_y[0, :]) ** 2)) * last_step_eff
        time_y = af.ifft(ffty * D * last_step / 2) * af.exp(atten) * af.exp(
            N * (af.abs(time_x[1, :]) ** 2 + af.abs(time_y[0, :]) ** 2)) * last_step_eff

        fftx = af.fft(time_x)
        ffty = af.fft(time_y)
        time_y = af.ifft(ffty * D * last_step / 2)
        time_x = af.ifft(fftx * D * last_step / 2)
        temp[0, :] = time_x.to_ndarray()
        temp[1, :] = time_y.to_ndarray()
        return temp

    def inplace_prop(self, signal):
        after_prop = self.prop(signal)
        signal.data_sample = after_prop

    @property
    def leff(self):
        return

    def __call__(self, signal):
        self.inplace_prop(signal)
