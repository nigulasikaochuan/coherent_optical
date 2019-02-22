from Base.SignalInterface import Signal
import numpy as np
class QualityMeter(object):

    @staticmethod
    def EVM():
        pass

    @staticmethod
    def SNR():
        pass

    @staticmethod
    def OSNR():
        pass

    @staticmethod
    def ANC():
        pass

    @staticmethod
    def PNC():
        pass

class Powermeter(object):

    @staticmethod
    def measure(wave_form):
        total_power = 0
        if isinstance(wave_form,Signal):
            for pol in range(wave_form.data_sample.shape[0]):
                total_power = total_power + np.mean(np.abs(wave_form.data_sample[pol,:])**2)

        elif isinstance(wave_form,np.ndarray):
            for pol in range(wave_form.shape[0]):
                total_power +=  np.mean(np.abs(wave_form[pol,:])**2)

        print(f'power is {total_power}w')
        print(f'power is {10*np.log10((total_power*1000)/1)}')

        return total_power, 10*np.log10(total_power * 1000)