import myenv
import numpy
import pandas
import gym
from datetime import datetime
from pandas.core import resample
from keras.models import model_from_json


# 行動が全て同じものになってしまう
# 原因 動かなければprofitは減ることはないから?
#      inputが正規化できていないから?


visible_bar = 32
read_index = 500


def ohlc2(self):
    return self.agg({'open': 'first',
                     'high': 'max',
                     'low': 'min',
                     'close': 'last'})
pandas.core.resample.DatetimeIndexResampler.ohlc2 = ohlc2

def get_data(file_path):
    # データ取得
    data = dict(pandas.read_json(file_path)['candles'])
    # DFに変形するために整形
    columns = ['time', 'open', 'high', 'low', 'close']
    d = dict()
    for c in columns:
        d[c] = []
    for i in range(len(data)):
        for c in columns:
            if c == columns[0]:
                d[c].append(data[i][c])
            else:
                d[c].append((data[i][c + 'Ask'] + data[i][c + 'Bid']) / 2)
    # DFに変換
    df = pandas.DataFrame.from_dict(d)
    # timeをpythonのdatetime型に変換
    df['time'] = df['time'].map(str_date_to_datetime_01)
    # timeをインデックスにセット
    df.set_index('time', inplace=True)
    return df

def make_obs(data, read_index):
    """
    observation 作成
    """
    # 本ステップ対象のobsを取得
    target = data.iloc[:read_index][data.columns]
    m1 = numpy.array(target.iloc[-1 * visible_bar:][target.columns])
    m5 = numpy.array(target.resample('5min').ohlc2().dropna().iloc[-1 * visible_bar:][target.columns])
    m15 = numpy.array(target.resample('15min').ohlc2().dropna().iloc[-1 * visible_bar:][target.columns])
    m30 = numpy.array(target.resample('30min').ohlc2().dropna().iloc[-1 * visible_bar:][target.columns])
    h1 = numpy.array(target.resample('1H').ohlc2().dropna().iloc[-1 * visible_bar:][target.columns])

    # 正規化(最後の1m closedataを基準とする)
    denom = m1[-1][0]
    m1 = m1 / denom
    m5 = m5 / denom
    m15 = m15 / denom
    m30 = m30 / denom
    h1 = h1 / denom

    return numpy.array([m1, m5, m15, m30, h1])

def str_date_to_datetime_01(str_date: str) -> datetime:
    """
    yyyy-mm-ddTHH:MM:SS.ffffffZ -> datetime
    :param str_date:
    :return:
    """
    return datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%fZ')



ENV_NAME = 'myenv-v3'
env = gym.make(ENV_NAME)
numpy.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = model_from_json(open('dqn_myenv-v3_model_v01.json').read())
model.load_weights('dqn_myenv-v3_weights_v01.h5f')

data = get_data("fx_data.json")


Episodes = 1


for _ in range(Episodes):

    done = False
    env.reset()
    env.data = data

    while not done:
        action = model.predict_classes(numpy.array([[make_obs(data, read_index)]]))

        observation, reward, done, info = env.step(action[0])
        read_index += 1

        if done:
            print('reward: ', reward)