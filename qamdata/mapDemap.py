from scipy.io import loadmat
import os
import numpy as np

dir = os.path.dirname(os.path.abspath(__file__))


def qammod(msg, order):
    msg = np.atleast_2d(msg)

    if order == 4:
        filename = '4qam'
    else:
        filename = str(order) + 'qam.mat'
    fullpath = os.path.join(dir, filename)
    symbol = loadmat(fullpath)['x'][0]
    res = []

    for message in msg[0]:
        res.append(symbol[message])

    return np.atleast_2d(np.array(res))


def qamdemod(symbol_recv, order, unitAveragePower=True):
    symbol_recv = np.atleast_2d(symbol_recv)
    msg = []
    decision_symbol = []
    if order == 4:
        filename = '4qam'
    else:
        filename = str(order) + 'qam.mat'
    fullpath = os.path.join(dir, filename)
    symbol = loadmat(fullpath)['x'][0]

    if not unitAveragePower:
        symbol_recv[0, :] = symbol_recv[0, :] / np.sqrt(np.mean(np.abs(symbol_recv[0, :]) ** 2))
    for sys in symbol_recv[0, :]:
        distance = np.abs(sys - symbol) ** 2
        minindex = np.argmin(distance)
        decision_symbol.append(symbol[minindex])
        msg.append(minindex)

    return decision_symbol, msg



def decision(symbols_in,constellation):
    '''

    :param symbols_in:  symbols to decision
    :param constellation: the constellation of signal
    :return: decision
    '''
    symbols_in = np.atleast_2d(symbols_in)
    constellation = np.atleast_2d(constellation)

    res = []
    for sym in symbols_in[0]:
        distance = np.abs(sym - constellation[0])
        minindex = np.argmin(distance)
        res.append(constellation[minindex])
        
    return np.array([res])