import numpy as np
import matplotlib.pyplot as plt
import visdom
class Signal:
    pass
def spectrum(x,Signal):
    pass

def eyediagram(x, sps, eyenumber=2, head=10):
    '''

    :param x: Signal object or numpy array
    :param sps: sample per symbol
    :param eyenumber: the number of eyediagram
    :param head: the number to cut at the beginning and the end of signal
    :return: None
    '''
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

            plt.show()
        else:
            plt.figure()
            plt.title('x_polarization')
            __plot_realeye(start_index, end_index, eyenumber, sps, x_xpol, plt)
            plt.figure()
            plt.title('y_polarization')
            __plot_realeye(start_index, end_index, eyenumber, sps, x_ypol, plt)
            plt.show()

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
            plt.show()

        else:
            plt.figure()
            __plot_realeye(start_index, end_index, eyenumber, sps, sig, plt)
            plt.show()


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
    y = np.array([x[0,:]+1j*np.real(x[0,:])],dtype=x.dtype)
    eyediagram(x, 40,5, 30)


if __name__ == '__main__':
    # x = np.array([[1 + 1j, 1 + 2j, 1 + 3j], [2 + 2j, 2 + 3j, 3 + 4j]])
    # y = np.array([np.real(x[0, :]), np.imag(x[0, :])])
    pass
    #
    # plt.plot(y[0, :])
    # plt.plot(y[1,:])
    # plt.show()

