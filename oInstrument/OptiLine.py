'''
    This file contains optical instrument along the signal's propagation
'''
import numpy as np
from numpy.fft import fftfreq
from scipy.constants import h, c

import arrayfire as af


class Brf:
    pass




class Edfa:

    def __init__(self, gain_db, nf, is_ase=True, mode='ConstantGain', expected_power=0):
        self.gain_db = gain_db
        self.nf = nf
        self.is_ase = is_ase
        self.mode = mode
        self.expected_power = expected_power

    def one_ase(self, signal):

        lamb = (2 * max(signal.lamb) * min(signal.lamb)) / (max(signal.lamb) + min(signal.lamb))
        One_ase = (h * c / lamb) * (self.gain_lin * 10 ^ (self.nf_lin / 10) - 1) / 2
        return One_ase

    @property
    def gain_lin(self):
        return 10 ** (self.gain_db / 10)

    @property
    def nf_lin(self):
        return 10 ** (self.nf / 10)

    def traverse(self, signal):
        if self.is_ase:
            noise = self.one_ase(signal) * signal.fs
        else:
            noise = 0
        if self.mode == 'ConstantGain':
            signal.data_sample = np.sqrt(self.gain_lin) * signal.data_sample + noise
            return
        if self.mode == 'ConstantPower':
            signal_power = np.mean(np.abs(signal.data_sample[0, :]) ** 2) + np.mean(
                np.abs(signal.data_sample[1, :]) ** 2)
            desired_power_linear = (10 ** (self.expected_power / 10)) / 1000
            linear_gain = desired_power_linear / signal_power
            signal.data_sample = np.sqrt(linear_gain) * signal.data_sample + noise


class LaserSource:
    pass


class IQ:
    pass


class MZM:

    def __init__(self, optical_signal, uper_elec_signal, lower_elec_signal, upbias, lowbias, vpidc=5, vpirf=5,
                 insertloss=6,
                 extinction_ratio=35):
        self.optical_signal = optical_signal
        self.uper_elec_signal = uper_elec_signal
        self.lower_elec_signal = lower_elec_signal
        self.upbias = upbias
        self.lowbias = lowbias

        self.vpidc = vpidc
        self.vpirf = vpirf
        self.insertloss = insertloss
        self.extinction_ratio = extinction_ratio

    def traverse(self):

        if self.extinction_ratio is not np.inf:
            ER = 10 ** (self.extinction_ratio / 10)
            eplio = np.sqrt(1 / ER)

        else:
            eplio = 0

        ysplit_upper = np.sqrt(0.5 + eplio)
        ysplit_lowwer = np.sqrt(0.5 - eplio)
        ubias = -self.vpidc / 2
        lbias = -ubias
        phi_upper = np.pi * self.uper_elec_signal / (self.vpirf) + np.pi * ubias / self.vpidc
        phi_lower = np.pi * self.lower_elec_signal / self.vpirf + np.pi * lbias / self.vpidc

        attuenation = 10 ** (self.insertloss / 10)

        h = ysplit_upper * np.exp(1j * phi_upper) + ysplit_lowwer * np.exp(1j * phi_lower)
        h = h / attuenation
        h = (1 / np.sqrt(2)) * h
        sample_out = h * self.optical_signal
        return sample_out


class IQ_modulator:

    def __init__(self, optical_signal, signal_datasample, iqratio=0, phase_bias=0, extinction_ratio=np.inf,
                 biase_mzm=(0, 0)
                 , insert_loss=6):
        '''

        :param signal: qam signal to modulate
        :param iqratio: the power ratio between I and Q arms (default = 0)
        :param phase_bias: the pase biase between two arms
        :param extinction_ratio: extinction ratio of the two nested modulators [dB]
        :param biase_mzm: bias of the two nested modulators (default = (0 0))
        '''

        self.inphase_signal = np.real(signal_datasample[0, :])
        self.quadarte_signal = np.imag(signal_datasample[1, :])

        self.iqratio = iqratio
        self.phasea_bias = phase_bias
        self.extinction_ratio = extinction_ratio
        self.biase_mzm = biase_mzm

        self.optical_signal = optical_signal
        self.insert_loss = insert_loss

    def modulate(self):
        fudu_ratio = 10 ** (self.iqratio / 20)
        i_ratio = fudu_ratio / (1 + fudu_ratio)
        q_ratio = 1 - i_ratio

        Ei = self.optical_signal * i_ratio
        Eq = self.optical_signal * q_ratio

        # create MZM
        mzmi = MZM(Ei, self.inphase_signal, -self.inphase_signal, self.biase_mzm[0], self.biase_mzm[1],
                   extinction_ratio=self.extinction_ratio
                   )
        mzmq = MZM(Eq, self.quadarte_signal, -self.quadarte_signal, self.biase_mzm[0], self.biase_mzm[1],
                   extinction_ratio=self.extinction_ratio
                   )
        isample = mzmi.traverse()
        qsample = mzmq.traverse()
        outsample = isample + qsample * np.exp(1j * np.pi / 2 + self.phasea_bias)

        return outsample


class optFilter:
    pass
