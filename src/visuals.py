import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
import pandas as pd
import seaborn as sns

def load(input_dir='../input'):
    currencies = {}

    currencies['bitcoin'] = pd.read_csv('{}/bitcoin_price.csv'.format(input_dir),
                                        parse_dates=['Date'], index_col=0)

    currencies['bitconnect'] = pd.read_csv('{}/bitconnect_price.csv'.format(input_dir),
                                           parse_dates=['Date'], index_col=0)

    currencies['dash'] = pd.read_csv('{}/dash_price.csv'.format(input_dir),
                                     parse_dates=['Date'], index_col=0)

    currencies['ethereum'] = pd.read_csv('{}/ethereum_price.csv'.format(input_dir),
                                         parse_dates=['Date'], index_col=0)

    currencies['iota'] = pd.read_csv('{}/iota_price.csv'.format(input_dir),
                                     parse_dates=['Date'], index_col=0)

    currencies['litecoin'] = pd.read_csv('{}/litecoin_price.csv'.format(input_dir),
                                         parse_dates=['Date'], index_col=0)

    currencies['monero'] = pd.read_csv('{}/monero_price.csv'.format(input_dir),
                                       parse_dates=['Date'], index_col=0)

    currencies['nem'] = pd.read_csv('{}/nem_price.csv'.format(input_dir),
                                    parse_dates=['Date'], index_col=0)

    currencies['neo'] = pd.read_csv('{}/neo_price.csv'.format(input_dir),
                                    parse_dates=['Date'], index_col=0)

    currencies['numeraire'] = pd.read_csv('{}/numeraire_price.csv'.format(input_dir),
                                          parse_dates=['Date'], index_col=0)

    currencies['omisego'] = pd.read_csv('{}/omisego_price.csv'.format(input_dir),
                                        parse_dates=['Date'], index_col=0)

    currencies['qtum'] = pd.read_csv('{}/qtum_price.csv'.format(input_dir),
                                     parse_dates=['Date'], index_col=0)

    currencies['ripple'] = pd.read_csv('{}/ripple_price.csv'.format(input_dir),
                                       parse_dates=['Date'], index_col=0)

    currencies['stratis'] = pd.read_csv('{}/stratis_price.csv'.format(input_dir),
                                        parse_dates=['Date'], index_col=0)

    currencies['waves'] = pd.read_csv('{}/waves_price.csv'.format(input_dir),
                                      parse_dates=['Date'], index_col=0)

    return currencies


def plot_candlestick(currencies, coin_type='bitcoin', coin_feat=['Close']):
    ohlc = currencies[coin_type][coin_feat].resample('10D').ohlc()
    ohlc.reset_index(inplace=True)
    ohlc['Date'] = ohlc['Date'].map(mdates.date2num)

    fig, ax = plt.subplots()

    candlestick_ohlc(ax, ohlc.values, width=2, colorup='g')
    ax.xaxis_date()

    plt.title('Candlestick Chart - {}'.format(coin_type))
    plt.xlabel('Time(Yr-M)')
    plt.ylabel('Value(USD)')
    plt.legend()
    plt.show()


def plot_correlation(type='spearman'):
    files_to_use = [
        'bitcoin_price.csv',
        'bitconnect_price.csv',
        'dash_price.csv',
        'ethereum_price.csv',
        'iota_price.csv',
        'litecoin_price.csv',
        'monero_price.csv',
        'nem_price.csv',
        'neo_price.csv',
        'numeraire_price.csv',
        'omisego_price.csv',
        'qtum_price.csv',
        'ripple_price.csv',
        'stratis_price.csv',
        'waves_price.csv']

    cols_to_use = []
    for ind, file_name in enumerate(files_to_use):
        currency_name = file_name.split('_')[0]
        if ind == 0:
            df = pd.read_csv('../input/' + file_name, usecols=['Date', 'Close'], parse_dates=['Date'])
            df.columns = ["Date", currency_name]
        else:
            temp_df = pd.read_csv('../input/' + file_name, usecols=['Date', 'Close'], parse_dates=['Date'])
            temp_df.columns = ['Date', currency_name]
            df = pd.merge(df, temp_df, on='Date')
        cols_to_use.append(currency_name)
    df.head()

    temp_df = df[cols_to_use]
    corrmat = temp_df.corr(method=type)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat, vmax=1., square=True)
    plt.title('{} correlation map'.format(type), fontsize=15)
    plt.show()


def plot_trend(currencies, coin_type='bitcoin', coin_feat=['Close']):
    plt.plot(currencies[coin_type][coin_feat])
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.xlabel('Time(Yr-M)')
    plt.ylabel('Value(USD)')
    plt.title('{} Price - {}'.format(coin_feat, coin_type))
    plt.show()