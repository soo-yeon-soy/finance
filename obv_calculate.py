'''
- Data : 2022.11.29
- Author : Jeong Soo Yeon
- Description :  주가 정보가 저장된 테이블 내 1년치 주가 정보를 로드하여
                 그 정보로 OBV 지수를 계산하여 매수/매도 타이밍인 종목들을 추천

'''
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
from sqlalchemy import create_engine
import sqlalchemy


def get_finance_db():
    finance_db = pymysql.connect(host = '', port=0, user='', passwd='', db='' )
    finance_cursor = finance_db.cursor()

    finance_cursor.execute('''
    select a.codeNum, b.codeName, a.`date`, a.closing, a.volume
    from t_finance a
    left join t_finance_code b
    on a.codeNum = b.codeNum 
    where a.`date` <= CURDATE() - INTERVAL 7 day and a.`date` >= CURDATE() - INTERVAL 373 day
    ''')
    # where a.`date` >= CURDATE() - INTERVAL 365 day

    finance_data = finance_cursor.fetchall()

    finance_df = pd.DataFrame(list(finance_data), columns=['codeNum', 'codeName', 'date', 'closing', 'volume'])

    # finance_cursor.execute('''
    #     select code as codeNum, RSI_14D
    #     from t_tech_company
    #     where opendate = (select max(opendate) from t_tech_company)
    #     and (RSI_14D <= 30 or RSI_14D >= 70) and RSI_14D != 0
    # ''')
    # tech_data = finance_cursor.fetchall()
    # tech_df = pd.DataFrame(list(tech_data), columns = ['codeNum', 'RSI'])

    return finance_df


def get_stock_list(df):
    df_cnt = df['codeNum'].value_counts()
    cnt_max = df_cnt[0]

    df_cnt_idx = df_cnt.index.tolist()
    df_cnt_val = df_cnt.values.tolist()

    stock_cd_cnt = pd.DataFrame(list(zip(df_cnt_idx, df_cnt_val)), columns = ['stock_cd','count'])
    stock_cd_cnt_fltr = stock_cd_cnt[stock_cd_cnt['count'] == cnt_max]
    stock_list = stock_cd_cnt_fltr['stock_cd'].tolist()

    return stock_list


# Buy/Sell signal Catch
# 매수 신호 : OBV > OBV_EMA  /  매도 신호 : OBV < OBV_EMA
# signal : signal 확인할 dataframe
# col1, col2 : OBV, OBV_EMA
def buy_sell(signal, col1, col2):
    sigPriceBuy, sigPriceSell = [], []
    flag = -1 # A flag for the trend upward/downward

    # Loop Through the length of the data set
    for i in range(0, len(signal)):
        # if OBV > OBV_EMA and flag != 1 then buy else sell
        if signal[col1][i] > signal[col2][i] and flag != 1:
            sigPriceBuy.append(signal['closing'][i])
            sigPriceSell.append(np.nan)
            flag = 1
        # else if OBV < OBV_EMA and flag != 0 then sell else buy
        elif signal[col1][i] < signal[col2][i] and flag != 0:
            sigPriceSell.append(signal['closing'][i])
            sigPriceBuy.append(np.nan)
            flag = 0

        # else OBV == OBV => append NaN
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return (sigPriceBuy, sigPriceSell)


# 단순 이동 평균 (Simple Moving Average, SMA)
def SMA(data, period=30, column='closing'):
    return data[column].rolling(window=period).mean()


# 지수 이동 평균 (Exponential Moving Average, EMA)
def EMA(data, period=20, column='closing'):
    return data[column].ewm(span=period, adjust=False).mean()


# 이동 평균 수렴/발산
#  - 단기 지수 이동 평균 : 12일 평균값으로 계산
#  - 장기 지수 이동 평균 : 26일 평균값으로 계산
#  - 신호선 : 9일 평균값으로 계산
def MACD(data, period_long=26, period_short=12, period_signal=9, column='closing'):
    shortEMA = EMA(data, period_short, column=column)
    longEMA = EMA(data, period_long, column=column)

    data['MACD'] = shortEMA - longEMA

    data['Signal_Line'] = EMA(data, period_signal, column='MACD')

    return data


# RSI 계산
def RSI(data, period=-14, column='closing'):
    delta = data[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    data['up'] = up
    data['down'] = down

    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))

    RS = AVG_Gain / AVG_Loss

    RSI = 100.0 - (100.0/(1.0 + RS))
    data['RSI'] = RSI

    return data


if __name__ == '__main__':
    start = time.time()
    finance_df= get_finance_db()
    # rsi30 = tech_df[tech_df['RSI'] <= 30] # 매수 타이밍
    # rsi70 = tech_df[tech_df['RSI'] >= 70] # 매도 타이밍

    # finance_df_part = finance_df[finance_df['codeNum'] == '086280']
    print('Getting Query data done!')
    # rsi_test = MACD(finance_df_part, period_long=26, period_short=12, period_signal=9)
    # rsi_test = RSI(rsi_test, period=14)
    # rsi_test['SMA'] = SMA(rsi_test, period=30)
    # rsi_test['EMA'] = EMA(rsi_test, period=20)
    # print(rsi_test)

    host = 'dev..kr'
    port = 0
    user = ''
    passwd = ''
    db_name = ''

    url = "mysql+pymysql://:@/?charset=utf8mb4"
    engine = create_engine(url)

    stock_list = get_stock_list(finance_df)

    buy_list, sell_list = [], []
    buy_df = pd.DataFrame()
    sell_df = pd.DataFrame()
    for i in stock_list:
        df = finance_df[finance_df['codeNum'] == i]
        # print(df.iloc[[-1]])
        stock_name = finance_df['codeName'][finance_df['codeNum'] == i].tolist()[0]
        df = df.reset_index()

        obv = [0]
        # obv.append(0)
        for j in range(1, len(df.closing)):
            # 종가가 전일 종가보다 클 때
            if df.closing[j] > df.closing[j - 1]:
                obv.append(obv[-1] + df.volume[j])
            # 종가가 전일 종가보다 작을 때
            elif df.closing[j] < df.closing[j - 1]:
                obv.append(obv[-1] - df.volume[j])
            else:
                obv.append(obv[-1])

        df['OBV'] = obv
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()

        x = buy_sell(df, 'OBV', 'OBV_EMA')

        df['Buy_Signal_Price'] = x[0]
        df['Sell_Signal_Price'] = x[1]
        df = df.fillna('nullValue')

        df = MACD(df, period_long=26, period_short=12, period_signal=9)
        df = RSI(df, period=14)
        df['SMA'] = SMA(df, period=30)
        df['EMA'] = EMA(df, period=20)

        # print(df.iloc[[-1]])
        # print(df['Sell_Signal_Price'][-1:].tolist())
        '''
        매도/매수 타이밍은 전날까지의 흐름을 파악해서, 전날 시세 대비 금일 시세가 떨어질 지, 오를 지를
        예측하는 것이라고 판단, 전날 데이터가 매도 타이밍인지, 매수 타이밍인지 체크
        '''

        if df['Sell_Signal_Price'][-1:].tolist() != ['nullValue'] \
                and df['Buy_Signal_Price'][-1:].tolist() == ['nullValue']:
            df_new = df.iloc[[-1]]
            sell_df = sell_df.append(df_new)

        elif df['Buy_Signal_Price'][-1:].tolist() != ['nullValue'] \
                and df['Sell_Signal_Price'][-1:].tolist() == ['nullValue']:
            df_new = df.iloc[[-1]]
            buy_df = buy_df.append(df_new)


    # print(sell_df.info())
    sell_df = sell_df[['codeNum', 'codeName', 'date', 'closing', 'volume', 'OBV', 'OBV_EMA','RSI', 'SMA','EMA']]
    sell_df['timing'] = '매도'
    buy_df = buy_df[['codeNum', 'codeName', 'date', 'closing', 'volume', 'OBV', 'OBV_EMA','RSI', 'SMA','EMA']]
    buy_df['timing'] = '매수'

    # print(sell_df)
    # print(buy_df)

    sell_df = sell_df[sell_df['RSI'] >= 70]
    buy_df = buy_df[(buy_df['RSI'] <= 30) & (buy_df['RSI'] > 0)]
    # print(type(buy_df))
    # print(type(sell_df))

    # print(sell_df)
    # print(buy_df)
    #
    # print(sell_df.info())
    # print(buy_df.info())

    total_df = pd.concat([sell_df, buy_df])
    total_df.to_csv('stock_prediction_0331.csv',index=False)
    # total_df.to_sql('m_obv_rsi_prediction', con=engine, if_exists='append', chunksize=1000, index=False,
    #                 dtype = {
    #                     'codeNum':sqlalchemy.types.VARCHAR(20),
    #                     'codeName':sqlalchemy.types.VARCHAR(100),
    #                     'date':sqlalchemy.types.VARCHAR(20),
    #                     'closing':sqlalchemy.types.INT,
    #                     'volume':sqlalchemy.types.BIGINT,
    #                     'OBV':sqlalchemy.types.BIGINT,
    #                     'OBV_EMA':sqlalchemy.types.FLOAT,
    #                     'RSI':sqlalchemy.types.FLOAT,
    #                     'SMA':sqlalchemy.types.FLOAT,
    #                     'EMA':sqlalchemy.types.FLOAT,
    #                     'timing':sqlalchemy.types.VARCHAR(10)
    #                 })

    # new_sell_df = pd.merge(sell_df, rsi70, how='inner', on='codeNum')
    # new_buy_df = pd.merge(buy_df, rsi30, how='inner', on='codeNum')
    #
    # print(new_sell_df)
    # print(new_buy_df)
    # print(buy_df.info())
    # print("########## 매수 ##############")
    # for b in buy_list:
    #     print(b)
    #
    # print("########## 매도 ##############")
    # for s in sell_list:
    #     print(s)

    end = time.time()
    print('Spent : ', end-start)