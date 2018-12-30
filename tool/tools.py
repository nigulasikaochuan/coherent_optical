import numpy as np

import visdom
import matplotlib

matplotlib.use('TKAGG')
import matplotlib.pyplot as plt  
class Signal:
    pass
def spectrum(x,Signal):
    pass

def eyediagram(x, sps, eyenumber=2, head=10,vis=None):
    '''

    :param x: Signal object or numpy array
    :param sps: sample per symbol
    :param eyenumber: the number of eyediagram
    :param head: the number to cut at the beginning and the end of signal
    :return: None
    '''
    if vis is None:
        vis = visdom.Visdom(env = 'eye diagram')
    if isinstance(x, Signal):
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
        if x_ypol.dtype in [np.complex,np.complex64,np.complex128]:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title('x-polarization')

            x_iq = np.array([np.real(x_xpol), np.imag(x_xpol)])

            __plot_realeye(start_index, end_index, eyenumber, sps, x_iq[0,:].reshape(1,-1), plt)
            plt.subplot(212)
            __plot_realeye(start_index, end_index, eyenumber, sps, x_iq[1,:].reshape(1,-1), plt)
            # plt.show()

            plt.figure()
            plt.subplot(211)
            plt.title('y-polarization')
            y_iq = np.array([np.real(x_ypol), np.imag(x_ypol)])
            __plot_realeye(start_index, end_index, eyenumber, sps, y_iq[0,:].reshape(1,-1), plt)
            plt.subplot(212)
            __plot_realeye(start_index, end_index, eyenumber, sps, y_iq[1,:].reshape(1,-1), plt)

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
        if sig.dtype in [np.complex64,np.complex]:
            plt.figure()
            sig_complex = np.array([np.real(sig), np.imag(sig)])
            plt.subplot(211)
            # plt.title('inphase')
            __plot_realeye(start_index, end_index, eyenumber, sps, sig_complex[0,:].reshape(1,-1), plt)
            plt.subplot(212)
            # plt.title('quaduarte phase')
            __plot_realeye(start_index, end_index, eyenumber, sps, sig_complex[1,:].reshape(1,-1), plt)
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
        plt_object.plot(inphase,color = 'dodgerblue',linestyle='-')
        # plt_object.hold()



def evaldelay(signal):
    pass

def plot_optical_filed(signal):
    pass

def ber2q(ber):
    pass






def main():
    class Signal:
        pass
    from scipy.io import loadmat
    head = 30
    x = loadmat('test.mat',mat_dtype=True)['x']

    print(x)
    y = np.array([x[0,:]+1j*np.real(x[0,:])],dtype=np.complex)

    eyediagram(np.array([y[0],y[0]]), 40,5, 30)


if __name__ == '__main__':
    main()