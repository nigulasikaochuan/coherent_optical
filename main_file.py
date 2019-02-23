from Base.SignalInterface import  Signal
from Base.SignalInterface import QamSignal
from eInstrument.ElectricInstrument import PulseShaping
from oInstrument.channel import LinearFiber






if __name__ == '__main__':
    config_data = {
        'signal1':dict(symbol_rate=35e9,mf="16-qam",symbol_length=2**16,sps=2,sps_in_fiber=14),
        'signal2':dict(symbol_rate=35e9,mf="16-qam",symbol_length=2**16,sps=2,sps_in_fiber=14),
        'signal3':dict(symbol_rate=35e9,mf="16-qam",symbol_length=2**16,sps=2,sps_in_fiber=14)
    }

    signal1 = QamSignal(**config_data['signal1'])
    signal12 = QamSignal(**config_data['signal2'])
    signal13 = QamSignal(**config_data['signal3'])

    config_data_shaping_filter = {
        "pulse_shaping": "rrc",
        "span": 1024,
        "sps": signal1.sps,
        'alpha':0.02,

    }

    shaping_filter = PulseShaping(**config_data_shaping_filter)
    shaping_filter(signal1)
