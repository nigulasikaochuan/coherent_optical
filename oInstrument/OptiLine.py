'''
    This file contains optical instrument along the signal's propagation
'''
import numpy as np
from scipy.constants import h, c
# from .. import Base
from Base.SignalInterface import QamSignal,Signal

import sys
print(sys.path)

class Edfa:


    def __init__(self, gain_db, nf, is_ase=True, mode='ConstantGain', expected_power=0):
        '''

        :param gain_db:
        :param nf:
        :param is_ase: 是否添加ase噪声
        :param mode: ConstantGain or ConstantPower
        :param expected_power: 当mode为ConstantPoower  时候，此参数有效
        '''

        self.gain_db = gain_db
        self.nf = nf
        self.is_ase = is_ase
        self.mode = mode
        self.expected_power = expected_power

    def one_ase(self, signal):
        '''

        :param signal:
        :return:
        '''

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

    def __init__(self,laser_power,line_width,is_phase_noise,center_frequence):
        '''

        :param laser_power: [dbm]
        :param line_width: [hz]
        :param is_phase_noise:[bool]
        '''
        self.laser_power = laser_power
        self.line_width = line_width
        self.is_phase_noise = is_phase_noise
        self.center_frequence = center_frequence

    @property
    def linear_laser_power(self):
        return (10**(self.laser_power/10))*0.001

    def __call__(self,signal:Signal):

        signal.signal_power = self.linear_laser_power
        signal._normalize_power()
        signal.data_sample_in_fiber = np.sqrt(self.linear_laser_power) * signal.data_sample_in_fiber
        signal.lamb = Signal.freq2lamb(self.center_frequence)
        if self.is_phase_noise:
            initial_phase = -np.pi + 2 * np.pi * np.random.randn(1)
            dtheta = np.sqrt(2*np.pi*1/signal.fs_in_fiber*self.line_width)*np.random.randn(1,signal.data_sample_in_fiber.shape[1])
            dtheta[0,0] = 0
            phase_noise = initial_phase + np.cumsum(dtheta,axis=1)

            signal[:] = signal.data_sample_in_fiber * np.exp(1j*phase_noise)

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

if __name__ == '__main__':
    symbol_rate = 35e9
    mf = '16-qam'
    signal_power = 0
    symbol_length = 2 ** 16
    sps = 2
    sps_infiber = 4


    parameter = dict(symbol_rate=symbol_rate, mf=mf, symbol_length=symbol_length, sps=sps,
                     sps_in_fiber=sps_infiber)

    signal = QamSignal(**parameter)
    signal[:] = np.array([[1, 2, 3, 4, 5]])

    laser = LaserSource(0,0.001,True,193.1e12)
    laser(signal)

