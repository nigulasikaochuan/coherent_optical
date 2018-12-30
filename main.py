'''
    仿真时候，采用离散的样本点，假设符号率为R GHZ,每个符号用sps个样点来表示。所以两个样点之间的时间间隔为
        1/R/sps/1e9 (s)
    采样频率为：
        fs = R*sps GHZ
    总的时间跨度为:
        range(0,(sps)*nsymbol*fs/N,fs/N)

'''


###
fiber_parameter = {
    'ssmf':{'alpha':0.2,"d":16.7,'gamma':1.3}
}
# frequence unit
Ghz = 1e9
hz = 1
Thz = 1e12
# distance unit
km = 1000
m = 1
nm = 1e-9

def reset_simulation_settings(config_file = None,**kwargs):
    if config_file:
        pass
    else:
        print('no config file, use kwargs to initializes simulation')
        #####signal parameter##########
        symbol_rates = kwargs['symbol_rates']
        sps = kwargs['sps']
        mfs = kwargs['mfs']
        if len(mfs) ==1:
            mfs = mfs*len(symbol_rates)

        signal_powers = kwargs['signal_powers']
        if len(signal_powers) == 1:
            signal_powers*=len(symbol_rates)
        #######span parameter############
        span_lengths = kwargs['span_lengths']
        span_kinds = kwargs['span_kinds']
        if len(span_kinds)==1:
            span_kinds*=len(span_lengths)
        span_alpha = []
        span_gamma = []
        span_d = []
        for span in span_kinds:
            span_alpha.append(fiber_parameter[span]['alpha'])
            span_gamma.append(fiber_parameter[span]['gamma'])
            span_d.append(fiber_parameter[span]['d'])
        ####################################
        print('-'*50)
        print('simulation parameters set finish')
        print('-'*50)


def get_sps_absolutefrequence(symbol_rates,center_frequence = 1550):
    '''
    :param center_frequence:中间信道的中心频率
    :param symbol_rates: 每个信道的符号速率，列表的长度代表了信道的个数
    :return: None

    功能描述：
        1.更新sps：根据符号速率更新sps,提供的sps不会使用，为了完整必须在配置文件中提供
        2.合理设置每个信道的符号长度
        3.根据中心信道的中心频率，得到每个信道的中心频率

    '''
    pass




if __name__ == '__main__':
    pass


