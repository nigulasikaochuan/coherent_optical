import numpy as np
from numpy.fft import fftfreq
from scipy.fftpack import fft, ifft
from scipy.signal import correlate
from scipy.signal import lfilter
from Base import SignalInterface

from qamdata.mapDemap import decision


class DispersionCompensation(object):
    '''
        using __call__ to comp cd

        if input is ndarray, then a new array will be returned

        if input is signal object,because reference is passed,the data behind it will change too. e.g. inplace
    '''

    def __call__(self, signal, spans, fs=None):

        freq_vector = fftfreq(signal.data_sample_in_fiber.shape[1], 1 / signal.fs_in_fiber)
        for i in range(signal.data_sample_in_fiber.shape[0]):
            signal.data_sample_in_fiber[i, :] = self.__comp(freq_vector, signal.data_sample_in_fiber[i, :], spans)

    def __comp(self, freq_vector, sample, spans):
        '''

        :param freq_vector: freq vecotr
        :param sample: [1,n] 2d-array
        :param span: Span object
        :return:
        no change input array
        '''

        freq_omeg = 2 * np.pi * freq_vector
        sample = np.atleast_2d(sample)
        for span in spans:
            beta2 = -span.beta2
            disper = (1j / 2) * beta2 * freq_omeg ** 2 * span.length
            for pol_index in range(sample.shape[0]):
                sample[pol_index, :] = ifft(fft(sample[pol_index, :]) * disper)

        return sample


class CMA(object):
    def __init__(self):
        pass


class CoherentFrontEnd(object):

    def __init__(self):
        pass


class SuperScalar(object):

    def __init__(self, block_length=200, D=4, g=0.15, filter_N=20):
        '''

        :param block_length:
        :param D: for pll
        :param g: for pll
        '''
        self.block_length = block_length

        self.D = D
        self.g = g
        self.pll = PLL(self.D, self.g)
        self.filter_N = filter_N

    def __call__(self, rx_signal, tx_signal, constl):
        symbol_rx = rx_signal.symbol
        symbol_tx = tx_signal.symbol

        assert symbol_rx.shape[0] == symbol_tx.shape[0]

        parallelization = symbol_rx.shape[1] / self.block_length
        estimate_phase = np.zeros((parallelization, 1))

        for pol_rx, pol_tx in zip(symbol_rx, symbol_tx):
            # cut of the last block
            pol_rx = pol_rx[0:len(pol_rx) - self.block_length]
            pol_tx = pol_tx[0:len(pol_tx) - self.block_length]

            pol_rx = np.reshape(pol_rx, (parallelization, self.block_length))

            pol_rx[0:-1:2, :] = np.fliplr(pol_rx[0:-1:2, :])

            pol_tx = np.reshape(pol_tx, (self.block_length, parallelization))

            pol_tx[0:-1:2, :] = np.fliplr(pol_rx[0:-1:2, :])

            estimate_phase[0:-1:2, :] = np.sum(
                pol_rx[0:-1:2, 0:2] / pol_tx[0:-1:2, 0:2] + pol_rx[1:-1:2, 0:2] / pol_rx[0:-1:2, 0:2])
            estimate_phase = np.angle(estimate_phase)

            estimate_phase[1:-1:2, :] = estimate_phase[0:-1:2, :]

            estimate_phase_mat = np.tile(estimate_phase, (1, self.block_length))

            pol_rx = pol_rx * np.exp(-1j * estimate_phase_mat)
            pol_rx = self.pll(pol_rx, constl)

            pol_rx[0:-1:2, :] = np.fliplr(pol_rx[0:-1:2, :])
            pol_rx = np.reshape(pol_rx, (1, -1))

            pol_rx_dec = decision(pol_rx, constl)
            h_phase_ml = pol_rx_dec / pol_rx
            b = np.ones((1, 2 * self.filter_N + 1))
            h_phase_ml_filter = lfilter(b[0, :], 1, h_phase_ml)
            h_phase_ml_filter = np.roll(h_phase_ml_filter, -self.filter_N)
            phase_ML = np.angle(h_phase_ml_filter)
            rx_Symbols_CPR = pol_rx * np.exp(-1j * phase_ML)

            symbol_rx.symbol = rx_Symbols_CPR

            # removing pilot symbols


class PLL(object):

    def __init__(self, d, g):
        self.d = d
        self.g = g

    def __call__(self, rx_signal, constl):

        if isinstance(rx_signal, SignalInterface.Signal):
            symbols = rx_signal.symbol
        else:
            symbols = rx_signal
            symbols = np.atleast_2d(symbols)

        error = np.zeros_like(symbols)
        decision_symbol = np.zeros_like(symbols)
        decision_symbol[:, 0] = symbols[:, 0]
        phase = np.zeros_like(symbols)

        cpr_symbols = np.zeros_like(symbols)
        for i in range(symbols.shape[0]):
            cpr_symbols[i, :] = symbols[i, :]

        for i in range(self.d + 1, symbols.shape[1]):
            decision_symbol[:, i - self.d] = decision(symbols[:, i - self.d], constl)
            error[:, i - self.d] = np.imag(cpr_symbols[:, i - self.d] * np.conj(decision_symbol[:, i - self.d]))
            phase[:, i - self.d + 1] = self.g * error[:, i - self.d] + phase[:, i - self.d]
            cpr_symbols[:, i] = symbols[:, i] * np.exp(-1j * phase[:, i - self.d + 1])

        return cpr_symbols


class SyncSignal(object):
    '''
            using __call__ to comp cd

            if input is ndarray, then a new array will be returned

            if input is signal object,because reference is passed,the data behind it will change too. e.g. inplace
    '''

    def __call__(self, signal, txsignal, sps=None):
        '''

        :param signal: rx ndarray samples or rx signal object
        :param txsignal: tx signal object
        :param sps: sps in receive, often 2
        :return:
        '''

        for i in range(signal.data_sample_in_fiber.shape[0]):
            signal.data_sample_in_fiber[i, :] = SyncSignal.syncsignal(txsignal.symbol[i, :],
                                                                      signal.data_sample_in_fiber[i, :],
                                                                      signal.sps_in_fiber)

    @staticmethod
    def syncsignal(symbol_tx, sample_rx, sps):
        '''

        :param symbol_tx: 发送符号
        :param sample_rx: 接收符号，会相对于发送符号而言存在滞后
        :param sps: samples per symbol
        :return: 收端符号移位之后的结果

        # 不会改变原信号

        '''
        symbol_tx = np.atleast_2d(symbol_tx)[0, :]
        sample_rx = np.atleast_2d(sample_rx)[0, :]

        res = correlate(sample_rx[::sps], symbol_tx)

        index = np.argmax(np.abs(res))

        out = np.roll(sample_rx, sps * (-index - 1 + symbol_tx.shape[0]))
        return np.atleast_2d(out)


class Lms_pll(object):
    '''
        lms input signal, the sps shoule be resampled to 2
    '''

    def __init__(self, ntaps, filter_choose, is_training, lr_ts=0.001, lr_td=0.01, is_plot=False, vis=None):
        self.is_training = is_training
        self.ntaps = ntaps
        self.lr_ts = lr_ts
        self.lr_dd = lr_td
        self.is_plot = is_plot
        if divmod(ntaps, 2)[1] == 0:
            ntaps += 1

        self.error_x = []
        self.error_y = []
        self.vis = vis

        self.hxx = np.zeros((1, self.ntaps)) + 1j * np.zeros((1, self.ntaps))
        self.hyy = np.zeros((1, self.ntaps)) + 1j * np.zeros((1, self.ntaps))
        self.hxy = np.zeros((1, self.ntaps)) + 1j * np.zeros((1, self.ntaps))
        self.hyx = np.zeros((1, self.ntaps)) + 1j * np.zeros((1, self.ntaps))
        if filter_choose == 0:
            self.hxx[0, int(self.ntaps / 2)] = 0
            self.hyy[0, int(self.ntaps / 2)] = 0
            self.hxy[0, int(self.ntaps / 2)] = 1
            self.hyx[0, int(self.ntaps / 2)] = 1
        elif filter_choose == 1:
            self.hxx[0, int(self.ntaps / 2)] = 1
            self.hyy[0, int(self.ntaps / 2)] = 1
            self.hxy[0, int(self.ntaps / 2)] = 0
            self.hyx[0, int(self.ntaps / 2)] = 0
        elif filter_choose == 2:
            self.hxx[0, int(self.ntaps) / 2] = 1
            self.hyy[0, int(self.ntaps) / 2] = 0
            self.hxy[0, int(self.ntaps) / 2] = 0
            self.hyx[0, int(self.ntaps) / 2] = 1
        elif filter_choose == 3:
            self.hxx[0, int(self.ntaps) / 2] = 0
            self.hyy[0, int(self.ntaps) / 2] = 1
            self.hxy[0, int(self.ntaps) / 2] = 1
            self.hyx[0, int(self.ntaps) / 2] = 0

    def prop(self, signal, pll=False, g=0.4, nloops=1):

        signal_xsample = signal.data_sample[0, :]
        signal_ysample = signal.data_sample[1, :]

        training_length = np.atleast_2d(signal.symbol).shape[1]
        trianing_symbol = np.roll(signal.symbol, shift=-int(np.floor(self.ntaps / 2 / 2)), axis=1)

        if not pll:
            g = 0
        else:
            g = g
            pll_phase_x = np.zeros((1, int(len(signal_xsample) / 2)))
            pll_phase_y = np.zeros((1, int(len(signal_xsample) / 2)))
            pll_error_x = np.zeros((1, int(len(signal_xsample) / 2)))
            pll_error_y = np.zeros((1, int(len(signal_xsample) / 2)))

        if self.is_training:

            lr = self.lr_ts
        else:
            lr = self.lr_dd
        # errors = np.zeros((len(signal_xsample) / 2, 1))
        errorsx = []
        errorsy = []
        hxxs = []
        hxys = []
        hyys = []
        hyxs = []
        signal_outx = np.zeros((1, int(len(signal_xsample) / 2)), dtype=np.complex)
        signal_outy = np.zeros((1, int(len(signal_xsample) / 2)), dtype=np.complex)

        for index in range(self.ntaps, len(signal_xsample) + 1, 2):
            xx = signal_xsample[index - self.ntaps: index]
            yy = signal_ysample[index - self.ntaps: index]

            xx = np.flip(xx)
            yy = np.flip(yy)

            xout_nopll = self.hxx @ xx.T + self.hxy @ yy.T
            yout_nopll = self.hyx @ xx.T + self.hyy @ yy.T

            if pll:
                xout = xout_nopll * np.exp(-1j * pll_phase_x[0, int((index - self.ntaps) / 2)])
                yout = yout_nopll * np.exp(-1j * pll_phase_y[0, int((index - self.ntaps) / 2)])
            else:
                xout = xout_nopll
                yout = yout_nopll

            signal_outx[0, int((index - self.ntaps) / 2)] = xout

            signal_outy[0, int((index - self.ntaps) / 2)] = yout

            if self.is_training and int((index - self.ntaps) / 2) < training_length:

                xout_cpr_decision = trianing_symbol[0, :][int((index - self.ntaps) / 2)]
                yout_cpr_decision = trianing_symbol[1, :][int((index - self.ntaps) / 2)]
            else:
                from qamdata.mapDemap import decision

                constl = signal.constl
                xout_cpr_decision = decision(xout, constl)
                yout_cpr_decision = decision(yout, constl)

            if pll:
                pll_error_x[0, int((index - self.ntaps) / 2)] = np.imag(xout * np.conj(xout_cpr_decision))
                pll_error_y[0, int((index - self.ntaps) / 2)] = np.imag(yout * np.conj(yout_cpr_decision))
                pll_phase_x[0, int((index - self.ntaps) / 2) + 1] = g * pll_error_x[0, int((index - self.ntaps) / 2)] + \
                                                                    pll_phase_x[0, int((index - self.ntaps) / 2)]
                pll_phase_y[0, int((index - self.ntaps) / 2) + 1] = g * pll_error_y[0, int((index - self.ntaps) / 2)] + \
                                                                    pll_phase_y[0, int((index - self.ntaps) / 2)]

            error_x = xout_cpr_decision - xout
            error_y = yout_cpr_decision - yout

            if pll:

                self.hxx = self.hxx + lr * error_x * np.conj(
                    xx * np.exp(-1j * pll_phase_x[int((index - self.ntaps) / 2)]))
                self.hxy = self.hxy + lr * error_x * np.conj(
                    yy * np.exp(-1j * pll_phase_y[int((index - self.ntaps) / 2)]))
                self.hyx = self.hyx + lr * error_y * np.conj(
                    xx * np.exp(-1j * pll_phase_x[int((index - self.ntaps) / 2)]))
                self.hyy = self.hyy + lr * error_y * np.conj(
                    yy * np.exp(-1j * pll_phase_y[int((index - self.ntaps) / 2)]))
            else:
                self.hxx = self.hxx + lr * error_x * np.conj(xx)
                self.hxy = self.hxy + lr * error_x * np.conj(yy)
                self.hyx = self.hyx + lr * error_y * np.conj(xx)
                self.hyy = self.hyy + lr * error_y * np.conj(yy)

            errorsx.append(abs(error_x) ** 2)
            errorsy.append(abs(error_y) ** 2)
            hxxs.append(self.hxx[0, 8])
            hxys.append(self.hxy[0, 8])
            hyxs.append(self.hyx[0, 8])
            hyys.append(self.hyy[0, 8])

        if self.is_plot:
            import matplotlib
            matplotlib.use('TKAGG')
            import matplotlib.pyplot as plt
            plt.subplot(211)

            plt.title('error in x-pol')
            plt.plot(errorsx)
            plt.subplot(212)
            # plt.title('error in y-pol')
            # plt.plot(errorsy)

            plt.plot(hxxs, 'r-', label='hxx')
            plt.plot(hxys, 'b-', label='hxy')
            plt.plot(hyys, 'k-', label='hyy')
            plt.plot(hyxs, 'm-', label='hyx')
            plt.legend()
            plt.show()

        return signal_outx, signal_outy


def rotate_nliphase(rx_signal, tx_signal):
    '''

    :param rx_signal: rx ndarray 1sps
    :param txsignal_signal: tx signal object
    :return: None

    inplace operation
    '''

    rx_symbol = rx_signal.symbol
    tx_symbol = tx_signal.symbol

    for i in range(rx_signal.shape[0]):
        rx_symbol = rx_symbol / np.sqrt(np.mean(np.abs(rx_symbol) ** 2))
        tx_symbol = tx_symbol / np.sqrt(np.mean(np.abs(tx_symbol) ** 2))
        average_phase = np.angle(np.sum(rx_symbol / tx_symbol))
        rx_symbol = rx_symbol * np.exp(-1j * average_phase)


def main():
    from scipy.io import loadmat
    import matplotlib
    matplotlib.use('TKAGG')
    import matplotlib.pyplot as plt

    x = loadmat('matlab.mat')['rxSamples']
    y = loadmat('txSymbols.mat')['txSymbols']

    # ntaps, filter_choose, is_training, lr_ts=0.01, lr_td=0.01
    timeSync = SyncSignal()
    after_sync_x = timeSync(y[0, :], x[0, :], 2)[0]
    after_sync_y = timeSync(y[1, :], x[1, :], 2)[0]

    after_sync = np.array([after_sync_x, after_sync_y])

    # after_sync = np.array([y[0], y[1]])
    class signal:
        pass

    signal.data_sample = after_sync
    signal.symbol = y
    lss_equalizer = Lms_pll(13, 1, True, is_plot=True)
    lss_equalizer.prop(signal)

    # print('hello world')


if __name__ == '__main__':
    # import matplotlib
    #
    # matplotlib.use('TKAGG')
    # import matplotlib.pyplot as plt
    # from scipy.io import  loadmat
    # signal = loadmat('s1.mat')['s1'][:,0]
    # signal2 = loadmat('s2.mat')['s2'][:,0]
    #
    # plt.subplot(311)
    # plt.plot(signal)
    # plt.subplot(312)
    # plt.plot(signal2)
    #
    # out = syncsignal(signal2,signal,1)
    # plt.subplot(313)
    # plt.plot(out)
    # plt.show()
    main()
