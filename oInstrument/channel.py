from Base import SignalInterface
import numpy as np


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
            self.D [ps/]
            self.length [km]

            self.beta2: caculate beta2 from D
        method:
            __call__: the signal will
    '''

    def __init__(self, alpha, D, length):
        self.alpha = alpha
        self.D = D
        self.length = length

    @property
    def beta2(self):
        pass

    def __prop(self, signal):
        pass

    def __inplace_prop(self, signal):
        data_samplex, data_sampley = self.__prop(signal)
        self.data_sample = np.array([data_samplex, data_sampley])

    def __call__(self, signal):
        self.__inplace_prop(signal)


class NonlinearFiber(LinearFiber):

    def __init__(self, alpha, D, length, gamma):
        super().__init__(alpha, D, length)
        self.gamma = gamma

    def __prop(self, signal):
        pass

    def __inplace_prop(self, signal):
        pass

    def __call__(self, signal):
        self.__inplace_prop(signal)
