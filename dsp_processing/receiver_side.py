import numpy as np
from numpy.fft import fftfreq
from scipy.fftpack import fft, ifft
from scipy.signal import correlate


class DispersionCompensation(object):
    '''
        using __call__ to comp cd

        if input is ndarray, then a new array will be returned

        if input is signal object,because reference is passed,the data behind it will change too. e.g. inplace
    '''

    def __call__(self, signal, span, fs=None):
        if isinstance(signal, np.ndarray):
            assert fs is not None
            temp = np.zeros_like(signal)
            freq_vector = fftfreq(signal.shape[1], 1 / fs)
            for i in range(signal.shape[0]):
                temp[i,:] = self.__comp(freq_vector,signal,span)
            return temp
        else:
            freq_vector = fftfreq(signal.data_sample.shape[1],1/signal.fs)
            for i in range(signal.data_sample.shape[0]):
                signal.data_sample[i,:] = self.__comp(freq_vector,  signal.data_sample[i,:],span)




    def __comp(self, freq_vector, sample, span):
        '''

        :param freq_vector: freq vecotr
        :param sample: [1,n] 2d-array
        :param span: Span object
        :return:

        no change input array
        '''
        freq_omeg = 2 * np.pi * freq_vector
        sample = np.atleast_2d(sample)
        beta2 = -span.beta2
        disper = (1j / 2) * beta2 * freq_omeg ** 2 * span.length
        fourier_transform = fft(sample[0,:])*disper
        after_comp = ifft(fourier_transform)

        return after_comp



class CMA(object):
    def __init__(self):
        pass


class CoherentFrontEnd(object):

    def __init__(self):
        pass



class SuperScalar(object):

    def __init__(self):
        pass



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
        if isinstance(signal, np.ndarray):
            assert sps is not None
            signal = np.atleast_2d(signal)
            temp = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                temp[i, :] = SyncSignal.syncsignal(txsignal.symbol[i, :], signal[i, :], sps)
            return temp
        else:
            for i in range(signal.data_sample[0]):
                signal.data_sample[i, :] = SyncSignal.syncsignal(txsignal.symbol[i, :], signal.data_sample[i, :],
                                                                 signal.sps)

    @staticmethod
    def syncsignal(symbol_tx, sample_rx, sps):
        '''

        :param symbol_tx: 发送符号
        :param sample_rx: 接收符号，会相对于发送符号而言存在滞后
        :param sps: samples per symbol
        :return: 收端符号移位之后的结果

        # 不会改变原信号

        '''
        symbol_tx = np.atleast_2d(symbol_tx)[0]
        sample_rx = np.atleast_2d(sample_rx)[0]

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


def rotate_nliphase(rx_signal,txsignal_signal):
    '''

    :param rx_signal: rx ndarray 1sps
    :param txsignal_signal: tx signal object
    :return: None

    inplace operation
    '''
    if isinstance(rx_signal,np.ndarray):
        # at_least 是view
        # ndarray 是可变对象，rx_signal 引用了外部对象，所以是inplace

        rx_signal = np.atleast_2d(rx_signal)
    else:
        # rx_signal 重新引用自己的symbol,inplace 操作,所以是inplace
        rx_signal = rx_signal.symbol
    refer_symbol = np.atleast_2d(txsignal_signal.symbol)


    pass
    ### add code to rotate phase shift symbol


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
