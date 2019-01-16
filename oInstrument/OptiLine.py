'''
    This file contains optical instrument along the signal's propagation
'''
import numpy as np
from numpy.fft import fftfreq
from scipy.constants import h, c

import arrayfire as af


class Brf:
    pass


class Fiber:
    '''
       alpha:[db/km] ,default:0.2
       length:[km] default:80
       D:Fiber dispesion coefficients [ps/nm/km] default:16.7
       S:Fiber dispesion slope [ps/nm^2/km] default:0
       gamma:   Fiber nonlinear coefficients [W^-1*km^-1] default:1.3
       lambda: the wavelength of signal[nm] ,defautlt:1550nm
    '''

    def __init__(self, alpha, length, D, S, gamma, lambda_signal, is_pmd=False, dgd=None, nplates=None, manakov=True,
                 step=5 / 1000):
        self.alpha = alpha
        self.length = length
        self.D = D
        self.S = S
        self.gamma = gamma
        self.lambda_signal = lambda_signal
        self.is_pmd = is_pmd
        self.step = step
        if self.is_pmd:
            assert dgd is not None
            assert nplates is not None
            if not manakov:
                raise NotImplementedError("still in progress.......")

            self.dgd = dgd
            self.nplates = nplates
        self.manakov = manakov

    def __checkstep(self, zprop, dz, lcorr, dz_miss, nz_old):

        nz = zprop / lcorr
        nz_index = np.ceil(nz)

        if dz_miss == 0:
            nmem = 0
            ntrunck = nz_index - nz_old
            dzlast = dz - lcorr * (ntrunck - 1)
            dzb = lcorr * np.ones((1, ntrunck))
            dzb[0, -1] = dzlast
        else:
            nmem = 1
            ntrunck = nz_index - nz_old + 1

            if ntrunck == 1:
                dzb = dz
                dz_miss = dz_miss - dz
            else:
                dzlast = dz - dz_miss - lcorr * (ntrunck - 2)
                dzb = lcorr * np.ones((1, ntrunck))
                dzb[0, 0] = dz_miss
                dzb[0, -1] = dzlast

        return dzb, dz_miss, nmem, ntrunck

    def __ssfm(self, signal):

        lambs = signal.lambs
        beta2 = self.beta2(lambs)
        dz_miss = 0
        x_pol = af.np_to_af_array(signal.data_sample[0, :])
        y_pol = af.np_to_af_array(signal.data_sample[1, :])
        freq = fftfreq(len(x_pol), 1 / signal.fs)
        freq = af.np_to_af_array(freq)
        omeg = np.pi * 2 * freq
        if self.manakov:
            self.gamma = 8 / 9 * self.gamma

        if self.is_pmd:
            db0 = np.random.rand(1, self.nplates) * 2 * np.pi - np.pi
            theta = np.random.rand(1, self.nplates) * np.pi - 0.5 * np.pi
            epsilon = 0.5 * np.arcsin(np.random.rand(1, self.nplates) * 2 - 1)
            dgdrms = np.sqrt((3 * np.pi) / 8) * self.dgd / np.sqrt(self.nplates)
            db1 = dgdrms * omeg

            Brf.db0 = db0
            Brf.theta = theta
            Brf.epsilon = Brf.epsilon

            wave_plate_length = self.length / self.nplates
            Brf.lcorr = wave_plate_length

            halfalpha = self.alphalin / 2
            waveplate_index = 0
            zprop = self.step
            while zprop < self.length:
                sigx, sigy = self.__nl_step(self.alphalin, self.gamma, self.step, x_pol, y_pol)
                ###check_step
                dzb, dz_miss, nmem, ntrunk = self.__checkstep(zprop, self.step, wave_plate_length, dz_miss,
                                                              waveplate_index)

                sigx, sigy = self.__linear_step(self.beta2(signal), db1, dzb, ntrunk, sigx, sigy, sig0, sig2, sig3i,
                                                Brf, waveplate_index, nmem)
                ############

                waveplate_index = waveplate_index + ntrunk - nmem
                sigx = sigx * np.exp(-halfalpha * self.step)
                sigy = sigy * np.exp(-halfalpha * self.step)
                zprop += self.step

            last_step = self.length - zprop + self.step

            sigx, sigy = self.__nl_step(self.alphalin, self.gamma, last_step, x_pol, y_pol)
            dzb, dz_miss, nmem, ntrunk = self.__checkstep(zprop, last_step, wave_plate_length, dz_miss, waveplate_index)

            sigx, sigy = self.__linear_step(self.beta2(signal), db1, dzb, ntrunk, sigx, sigy, sig0, sig2, sig3i, Brf,
                                            waveplate_index, nmem)
            sigx = sigx * np.exp(-halfalpha * last_step)
            sigy = sigy * np.exp(-halfalpha * last_step)

            Brf.betat = beta2
            Brf.db1 = db1
            return sigx, sigy

    def __nl_step(self, alphalin, gamma, step, x_pol, y_pol):
        leff = (1 - np.exp(-alphalin * step)) / alphalin
        signal_power = np.abs(x_pol) ** 2 + np.abs(y_pol) ** 2
        gamma_leff = self.gamma * leff
        nlinear = np.exp(-gamma_leff * signal_power)
        tempx = x_pol * nlinear
        tempy = y_pol * nlinear
        return tempx, tempy

    @property
    def beta3(self):
        print("waring, the beta3 is set to 0 暂时")
        return 0

    @property
    def leff(self):
        return (1 - np.exp(-self.alphalin * self.length)) / self.alphalin

    @property
    def alphalin(self):

        return np.log(10 ** (self.alpha / 10))

    def beta2(self, lamb):
        '''

        :param lamb: [nm] the signal's lambdas
        :return: beta2 of this channel
        '''
        lamb = (2 * max(lamb) * min(lamb)) / (max(lamb) + min(lamb))

        return -self.D * (lamb * 1e-12) ^ 2 / (2 * np.pi * c * 1e-3)  # s^2/km

    def traverse(self, signal):
        self.__ssfm(signal)

    def __linear_step(self, beta2, db1, dzb, ntrunk, sigx, sigy, sig0, sig2, sig3i, brf, ntot, nmem):
        ux = af.fft(sigx)
        uy = af.fft(sigy)

        for k in range(0, int(ntrunk)):
            n = ntot + k - nmem
            matRth = np.cos(brf.theta[n]) * sig0 - np.sin(brf.theta[n]) * sig3i
            matRepsilon = np.cos(brf.epsilon[n]) * sig0 + 1j * np.sin(brf.epsilon[n]) * sig2
            matR = matRth @ matRepsilon

            uux = np.conj(matR[0, 0]) * ux + np.conj(matR[1, 0]) * uy
            uuy = np.conj(matR[0, 1]) * ux + np.conj(matR[1, 1]) * uy

            combeta = beta2 * dzb[k]
            deltabeta = 0.5 * (db1 + brf.db0[n]) * dzb[k] / brf.lcorr
            uux = np.exp(-(combeta + deltabeta)) * uux
            uuy = np.exp(-(combeta - deltabeta)) * uuy

            ux = matR[0, 0] * uux + matR[0, 1] * uuy
            uy = matR[1, 0] * uux + matR[1, 1] * uuy

        ux = af.ifft(ux)
        uy = af.ifft(uy)
        return ux, uy


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
