import numpy as np
from scipy.signal import welch
import visdom
import matplotlib

matplotlib.use('TKAGG')
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import tools
from Base import SignalInterface
import visdom
from scipy.spatial import Voronoi
from scipy.interpolate import griddata
from scipy.signal import convolve
from scipy.signal import lfilter

try:
    import matlab
    import matlab.engine as engine

    eng = engine.connect_matlab()
except Exception as e:
    print('')


class SpectrumAnalyzer(object):

    def __init__(self, window='hamming', ratio=1):
        if window == 'hamming':
            self.window = np.hamming(1024 * ratio)
        if window == 'hanning':
            self.window = np.hanning(1024 * ratio)

        self.bandwidth_3db = [0, 0]
        self.bandwidth_6db = [0, 0]
        self.bandwidth_10db = [0, 0]

    def __call__(self, signal, env="new env", fs=None, server="http://192.168.0.76", port=8097):
        if isinstance(signal, SignalInterface.Signal):
            x = signal.data_sample[0]
            y = signal.data_sample[1]
            self.__plot(x, y, signal.fs, env, server, port)
        else:
            assert fs is not None
            signal = np.atleast_2d(signal)
            x = signal[0]
            y = signal[1]
            self.__plot(x, y, fs, env, server, port)

    def __plot(self, x, y, fs, env, server, port):
        f, pxx = welch(x, fs, self.window, return_onesided=False)
        f2, px2 = welch(y, fs, self.window, return_onesided=False)

        pxx = 10 * np.log10(pxx)
        px2 = 10 * np.log10(px2)

        viz = visdom.Visdom(server="http://192.168.0.176", port=8097, env=env)
        fig = tools.make_subplots(rows=2, cols=1, subplot_titles=['x_pol power_spectrum', 'y_pol power_spectrum'])
        specx = go.Scattergl(x=f / 1e9, y=pxx, mode='lines', name='power density spectrum')
        specy = go.Scattergl(x=f2 / 1e9, y=px2, mode="lines", name='power density spectrum')
        fig.append_trace(specx, 1, 1)
        fig.append_trace(specy, 2, 1)

        fig['layout']['xaxis1'].update(title='frequence[GHz]', exponentformat='e', showexponent='all', showline=True)
        fig['layout']['xaxis2'].update(title='frequence[GHz]', exponentformat='e', showexponent='all', showline=True)

        fig['layout']['yaxis1'].update(title='power spectrum density [W/Hz] dB', showgrid=True, gridcolor='#bdbdbd')
        fig['layout']['yaxis2'].update(title='power spectrum density [W/Hz] dB', showgrid=True, gridcolor='#bdbdbd')

        fig['layout']['xaxis1'].update(showgrid=True, gridcolor='#bdbdbd')
        fig['layout']['xaxis2'].update(showgrid=True, gridcolor='#bdbdbd')
        viz.plotlyplot(fig)

        bw_3 = pxx >= (np.max(pxx) - 3)
        cutoff_3 = max(f[bw_3])
        self.bandwidth_3db[0] = cutoff_3
        bw_3 = pxx >= (np.max(px2) - 3)
        cutoff_3 = max(f[bw_3])
        self.bandwidth_3db[1] = cutoff_3


class Scatterplot(object):

    def __call__(self, signal):
        if isinstance(signal, SignalInterface.Signal):
            x = signal.data_sample[0]
            y = signal.data_sample[1]

        else:
            signal = np.atleast_2d(signal)
            x = signal[0]
            y = signal[1]

        self.__scatterplot(x, y)

    def __scatterplot(self, x, y):
        pass





def eyediagram(x, sps, eyenumber=2, head=10, vis=None):
    '''

    :param x: Signal object or numpy array
    :param sps: sample per symbol
    :param eyenumber: the number of eyediagram
    :param head: the number to cut at the beginning and the end of signal
    :return: None
    '''
    if vis is None:
        vis = visdom.Visdom(env='eye diagram')
    if isinstance(x, SignalInterface.Signal):
        x = x.data_sample
    else:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
    start_index = head
    end_index = (np.floor(x.shape[1] / sps)) - head

    if x.shape[0] == 2:
        print('polarization demultiplexed signal')

        x_xpol = x[0, :]  # x 方向的信号
        x_xpol.shape = 1, -1
        x_ypol = x[1, :]  # y 方向的信号
        x_ypol.shape = 1, -1
        if x_ypol.dtype in [np.complex, np.complex64, np.complex128]:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title('x-polarization')

            x_iq = np.array([np.real(x_xpol), np.imag(x_xpol)])

            __plot_realeye(start_index, end_index, eyenumber, sps, x_iq[0, :].reshape(1, -1), plt)
            plt.subplot(212)
            __plot_realeye(start_index, end_index, eyenumber, sps, x_iq[1, :].reshape(1, -1), plt)
            # plt.show()

            plt.figure()
            plt.subplot(211)
            plt.title('y-polarization')
            y_iq = np.array([np.real(x_ypol), np.imag(x_ypol)])
            __plot_realeye(start_index, end_index, eyenumber, sps, y_iq[0, :].reshape(1, -1), plt)
            plt.subplot(212)
            __plot_realeye(start_index, end_index, eyenumber, sps, y_iq[1, :].reshape(1, -1), plt)

            vis.matplot(plt)
        else:
            plt.figure()
            plt.title('x_polarization')
            __plot_realeye(start_index, end_index, eyenumber, sps, x_xpol, plt)
            plt.figure()
            plt.title('y_polarization')
            __plot_realeye(start_index, end_index, eyenumber, sps, x_ypol, plt)
            vis.matplot(plt)

    else:
        print('one polarization')
        sig = x[0, :]
        sig.shape = 1, -1
        if sig.dtype in [np.complex64, np.complex]:
            plt.figure()
            sig_complex = np.array([np.real(sig), np.imag(sig)])
            plt.subplot(211)
            # plt.title('inphase')
            __plot_realeye(start_index, end_index, eyenumber, sps, sig_complex[0, :].reshape(1, -1), plt)
            plt.subplot(212)
            # plt.title('quaduarte phase')
            __plot_realeye(start_index, end_index, eyenumber, sps, sig_complex[1, :].reshape(1, -1), plt)
            vis.matplot(plt)

        else:
            plt.figure()
            __plot_realeye(start_index, end_index, eyenumber, sps, sig, plt)
            vis.matplot(plt)


def __plot_realeye(start_index, end_index, eyenumber, sps, signal, plt_object):
    start_index = int(start_index)
    end_index = int(end_index)
    for index in range(start_index, end_index):
        inphase = signal[0, index * sps + 1:(index + eyenumber) * sps + 1]
        plt_object.plot(inphase, color='dodgerblue', linestyle='-')
        # plt_object.hold()


def evaldelay(signal):
    pass



def ber2q(ber):
    pass


def main():
    from scipy.io import loadmat

    x = loadmat('/Volumes/D/zaxiang/mysimulation/gitcode/dsp_processing/matlab.mat')['rxSamples']
    y = x[0]
    y = dowmsample(y, 2)
    y2 = x[1]
    y2 = dowmsample(y2, 2)
    # spectrum = SpectrumAnalyzer()
    # spectrum(x, fs=35e9 * 2)

    scatter = Scatterplot()
    scatter(np.array([y, y2]))


def dowmsample(x, sps):
    return x[0:len(x):sps]


if __name__ == '__main__':
    main()
