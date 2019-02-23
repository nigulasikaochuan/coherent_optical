from Base import SignalInterface
from scipy.signal import convolve
from scipy.signal import resample
import numpy as np


class ADC(object):

    def __init__(self, sps):
        self.sps = sps

    def __call__(self, signal):
        pass


class DAC(object):

    def __call__(self, signal):
        sps_in_fiber = signal.sps_in_fiber
        N = sps_in_fiber / signal.sps * signal.symbol_length
        tempx = resample(signal.data_sample_in_fiber[0, :], N)
        tempy = resample(signal.data_sample_in_fiber[1, :], N)
        signal.data_sample_in_fiber = np.array([tempx, tempy])


class PulseShaping(object):

    def __init__(self, **kwargs):
        self.pulse_shaping = kwargs['pulse_shaping']
        self.span = kwargs['span']
        self.sps = kwargs['sps']
        self.alpha = kwargs['alpha']
        assert divmod(self.span * self.sps, 2)[1] == 0
        self.number_of_sample = self.span * self.sps
        self.delay = self.span / 2 * self.sps

        self.filter_tap = self.__design_filter()

    def __design_filter(self):
        if self.pulse_shaping == 'rrc':
            h = PulseShaping.rcosdesign(self.number_of_sample, self.alpha, 1, self.sps)
            return np.atleast_2d(h)

        if self.pulse_shaping == 'rc':
            print(
                'error, why do you want to design rc filter,the practical use should be two rrc filters,on in transimit'
                'side,and one in receiver side, rrc filter will be designed')

    @staticmethod
    def rcosdesign(N, alpha, Ts, Fs):
        """
        Generates a root raised cosine (RRC) filter (FIR) impulse response.
        Parameters
        ----------
        N : int
            Length of the filter in samples.
        alpha : float
            Roll off factor (Valid values are [0, 1]).
        Ts : float
            Symbol period in seconds.
        Fs : float
            Sampling Rate in Hz.
        Returns
        ---------
        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for
            the impulse response.
        h_rrc : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.
        """

        T_delta = 1 / float(Fs)

        sample_num = np.arange(N + 1)
        h_rrc = np.zeros(N + 1, dtype=float)

        for x in sample_num:
            t = (x - N / 2) * T_delta
            if t == 0.0:
                h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
            elif alpha != 0 and t == Ts / (4 * alpha):
                h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                    (np.sin(np.pi / (4 * alpha)))) + (
                                                           (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
            elif alpha != 0 and t == -Ts / (4 * alpha):
                h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                    (np.sin(np.pi / (4 * alpha)))) + (
                                                           (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
            else:
                h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) + \
                            4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                           (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

        return h_rrc / np.sqrt(np.sum(h_rrc * h_rrc))

    def rrcfilter(self, signal_interface):
        '''

        :param signal_interface: signal object to be pulse shaping,because a reference of signal object is passed
        so the filter is in place
        :return: None

        '''

        print("---begin pulseshaping ---")
        # upsample by insert zeros



        for i in range(signal_interface.data_sample.shape[0]):

            signal_interface.data_sample[i, :] = signal_interface.upsample(signal_interface.symbol[i, :],signal_interface.sps)[0, :]


        temp =[]
        for i in range(signal_interface.data_sample.shape[0]):
            temp.append(convolve(self.filter_tap[0, :], signal_interface.data_sample[i, :]))

        # tempy = convolve(self.filter_tap[0, :], signal_interface.data_sample[1, :])
        # temp_signal = np.array([tempx, tempy])
        temp_signal = np.array(temp)
        # compensate group delay
        temp_signal = np.roll(temp_signal, -int(self.delay), axis=1)

        signal_interface.data_sample = temp_signal[:, :signal_interface.sps * signal_interface.symbol_length]

        print('rrc filter completed')

    def __call__(self, signal):
        self.rrcfilter(signal)


class AWG(object):

    def __init__(self, pulse_shaping_dict, signal_interface: SignalInterface.QamSignal):
        '''

        :param pulse_shaping_dict: pulse shaping parameters
        :param signal_interface: reference of the signal object,change signal_interface's property will change all
        '''

        self.signal_interface = signal_interface
        pulse_shaping_dict['sps'] = self.signal_interface.sps

        self.pulse_shaping_filter = PulseShaping(**pulse_shaping_dict)

    def __call__(self, *args, **kwargs):
        self.pulse_shaping_filter.rrcfilter(self.signal_interface)


if __name__ == '__main__':
    h = PulseShaping.rcosdesign(32 * 4, 0.2, 1, 4)
