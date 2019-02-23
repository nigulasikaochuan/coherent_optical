import sys

sys.path.append('../')
from numpy.fft import fftfreq
from scipy.fftpack import fft, ifft
try:
    import cupy as cp
    from cupy.fft import fft as cfft
    from cupy.fft import ifft as icfft
except Exception as e:
    print('cupy can not be used')

from Base import SignalInterface
import numpy as np
from scipy.constants import c

import math


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
    def alpha_lin(self):
        return np.log(10 ** (self.alpha / 10))

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
        return (1 - np.exp(-self.alpha_lin * self.step_length)) / self.alpha_lin

    def cupy_prop(self, signal):

        flag = False

        steps = self.length / self.step_length
        steps = np.floor(steps)
        steps = int(steps)
        if steps * self.step_length < self.length:
            last_step = self.length - steps * self.step_length
            flag = True

        xpol = cp.asarray(signal.data_sample[0, :], dtype=cp.complex)
        ypol = cp.asarray(signal.data_sample[1, :], dtype=cp.complex)

        freq = fftfreq(len(xpol), 1 / signal.fs)
        freq = cp.asarray(freq)
        omeg = 2 * np.pi * freq

        linear_operator = 1j / 2 * self.beta2 * omeg ** 2

        nonlinear_operator = self.gamma * self.step_length_eff * 8 / 9 * 1j

        for i in range(steps):
            print(i)

            xpol_fft = cfft(xpol)
            xpol_fft = xpol_fft * cp.exp(linear_operator * self.step_length / 2)

            ypol_fft = cfft(ypol)
            ypol_fft = ypol_fft * cp.exp(linear_operator * self.step_length / 2)

            xpol = icfft(xpol_fft)
            ypol = icfft(ypol_fft)

            xpol = xpol * cp.exp(-nonlinear_operator * (cp.abs(xpol) ** 2 + cp.abs(ypol) ** 2))
            ypol = xpol * cp.exp(-nonlinear_operator * (cp.abs(xpol) ** 2 + cp.abs(ypol) ** 2))

            xpol = xpol * cp.exp(- self.alpha_lin * self.step_length / 2)
            ypol = ypol * cp.exp(- self.alpha_lin * self.step_length / 2)

            xpol_fft = cfft(xpol)
            xpol_fft = xpol_fft * cp.exp(linear_operator * self.step_length / 2)

            ypol_fft = cfft(ypol)
            ypol_fft = ypol_fft * cp.exp(linear_operator * self.step_length / 2)

            xpol = icfft(xpol_fft)
            ypol = icfft(ypol_fft)

        if flag:
            step_size_eff = 1 - cp.exp(-last_step * self.alpha_lin)
            step_size_eff = step_size_eff / self.alpha_lin
            nonlinear_operator = self.gamma * step_size_eff * 8 / 9 * 1j
            xpol_fft = cfft(xpol)
            xpol_fft = xpol_fft * cp.exp(linear_operator * last_step / 2)

            ypol_fft = cfft(ypol)
            ypol_fft = ypol_fft * cp.exp(linear_operator * last_step / 2)

            xpol = icfft(xpol_fft)
            ypol = icfft(ypol_fft)

            xpol = xpol * cp.exp(nonlinear_operator * (cp.abs(xpol) ** 2 + cp.abs(ypol) ** 2))
            ypol = xpol * cp.exp(nonlinear_operator * (cp.abs(xpol) ** 2 + cp.abs(ypol) ** 2))
            xpol = xpol * cp.exp(- self.alpha_lin * last_step / 2)
            ypol = ypol * cp.exp(- self.alpha_lin * last_step / 2)

            xpol_fft = cfft(xpol)
            xpol_fft = xpol_fft * np.exp(linear_operator) * last_step / 2

            ypol_fft = cfft(ypol)
            ypol_fft = ypol_fft * np.exp(linear_operator) * last_step / 2

            xpol = icfft(xpol_fft)
            ypol = icfft(ypol_fft)

        xpol_numpy = cp.asnumpy(xpol)
        ypol_numpy = cp.asnumpy(ypol)

        return np.array([xpol_numpy, ypol_numpy])

    def arrayfire_prop(self, signal):
        print('arrayfire is used')
        if signal.data_sample.ndim != 2:
            raise Exception("only dp signal supported at this time")
        step_number = self.length / self.step_length
        step_number = int(np.floor(step_number))
        temp = np.zeros_like(signal.data_sample)
        freq = fftfreq(signal.data_sample.shape[1], 1 / signal.fs)
        freq_gpu = af.np_to_af_array(freq)
        omeg = 2 * np.pi * freq_gpu
        D = 1j / 2 * self.beta2 * omeg ** 2
        N = 8 / 9 * 1j * self.gamma
        atten = -self.alpha_lin / 2

        time_x = af.np_to_af_array(signal.data_sample[0, :])
        time_y = af.np_to_af_array(signal.data_sample[1, :])

        print('*' * 20 + "begin simulation" + '*' * 20)
        for i in range(step_number):
            print(i)
            fftx = af.fft(time_x)
            ffty = af.fft(time_y)

            time_x = af.ifft(fftx * af.exp(D * self.step_length / 2))
            time_y = af.ifft(ffty * af.exp(D * self.step_length / 2))
            time_x = time_x * af.exp(
                N * self.step_length_eff * (af.abs(time_x) ** 2 + af.abs(time_y) ** 2))
            time_y = time_y * af.exp(
                N * self.step_length_eff * (af.abs(time_x) ** 2 + af.abs(time_y) ** 2))

            time_x = time_x * math.exp(atten * self.step_length)
            time_y = time_y * math.exp(atten * self.step_length)
            fftx = af.fft(time_x)
            ffty = af.fft(time_y)
            time_y = af.ifft(ffty * af.exp(D * self.step_length / 2))
            time_x = af.ifft(fftx * af.exp(D * self.step_length / 2))

        print('*' * 20 + "end simulation" + '*' * 20)

        last_step = self.length - self.step_length * step_number
        last_step_eff = (1 - np.exp(-self.alpha * last_step)) / self.alpha
        if last_step == 0:
            temp[0, :] = time_x.to_ndarray()
            temp[1, :] = time_y.to_ndarray()
            return temp

        fftx = af.fft(time_x)
        ffty = af.fft(time_y)

        time_x = af.ifft(fftx * af.exp(D * last_step / 2))
        time_y = af.ifft(ffty * af.exp(D * last_step / 2))
        time_x = time_x * af.exp(
            N * last_step_eff * (af.abs(time_x) ** 2 + af.abs(time_y) ** 2))
        time_y = time_y * af.exp(
            N * last_step_eff * (af.abs(time_x) ** 2 + af.abs(time_y) ** 2))

        time_x = math.exp(atten * self.step_length) * time_x
        time_y = math.exp(atten * self.step_length) * time_y

        fftx = af.fft(time_x)
        ffty = af.fft(time_y)
        time_y = af.ifft(ffty * D * last_step / 2)
        time_x = af.ifft(fftx * D * last_step / 2)
        temp[0, :] = time_x.to_ndarray()
        temp[1, :] = time_y.to_ndarray()
        return temp

    def inplace_prop(self, signal):
        after_prop = self.arrayfire_prop(signal)
        signal.data_sample = after_prop

    @property
    def leff(self):
        return

    def __call__(self, signal):
        self.inplace_prop(signal)


if __name__ == "__main__":
    from scipy.io import loadmat
    from tool import qualitymeter
    import time

    samples = loadmat(r'C:\Users\lun\Desktop\coherent_optical-master\oInstrument\matlab.mat')['sample']
    span = NonlinearFiber(alpha=0.22, D=16.89, gamma=1.3, length=80, step_length=10e-3)


    class Signal(object):
        pass


    signal = Signal()
    signal.fs = 2.736000000000000e+11
    signal.data_sample = samples
    qualitymeter.Powermeter.measure(signal.data_sample)
    now = time.time()
    span(signal)
    print(time.time() - now)
    qualitymeter.Powermeter.measure(signal.data_sample)
