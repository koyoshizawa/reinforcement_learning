"""
https://github.com/hide-tono/keras-rl-fx/blob/master/envs/fx_env/fx_env.py 参考
"""

import gym
import numpy
import pandas
from datetime import datetime


class MyEnv(gym.Env):

    def __init__(self):

        self.HOLD = 0
        self.BUY = 1
        self.SELL = 2
        self.CLOSE = 3

        self.initial_balance = 1000000

        self.file_path = 'fx_data.json'
        self.read_index = 200
        self.visible_bar = 32
        self.data = self.get_data()

        # スプレッド
        self.spread = 0.5
        # Point(1pipsの値)
        self.point = 0.01
        # 利食いpips
        self.take_profit_pips = 30
        # 損切りpips
        self.stop_loss_pips = 15
        # ロット数
        self.lots = 10000

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=numpy.shape(self.make_obs('ohlc_array')))

    def _seed(self, seed=None):
        pass

    def _close(self):
        pass

    def get_data(self):
        # データ取得
        data = dict(pandas.read_json(self.file_path)['candles'])
        # DFに変形するために整形
        columns = ['time','open', 'high', 'low', 'close']
        d = dict()
        for c in columns:
            d[c] = []
        for i in range(len(data)):
            for c in columns:
                if c == columns[0]:
                    d[c].append(data[i][c])
                else:
                    d[c].append((data[i][c+'Ask'] + data[i][c+'Bid'])/2)
        # DFに変換
        df = pandas.DataFrame.from_dict(d)
        # timeをpythonのdatetime型に変換
        df['time'] = df['time'].map(str_date_to_datetime_01)
        # timeをインデックスにセット
        df.set_index('time', inplace=True)
        return df


    def _reset(self):
        # 口座情報初期化
        self.info = AccountInformation(self.initial_balance)

        self.read_index = 200
        self.data = self.get_data()

        # チケット一覧
        self.tickets = []

        return self.make_obs('ohlc_array')


    def _step(self, action):

        current_data = self.data.iloc[self.read_index]
        ask = current_data['close'] + self.spread * self.point
        bid = current_data['close'] - self.spread * self.point

        if action == self.HOLD:
            for ticket in self.tickets:
                if ticket.order_type == self.BUY:
                    if bid > ticket.take_profit:
                        # 買いチケットを利確
                        profIt = (ticket.take_profit - ticket.open_price) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_buy += profit
                    elif bid < ticket.stop_loss:
                        # 買いチケットを損切り
                        profit = (ticket.stop_loss - ticket.open_price) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_buy += profit
                elif ticket.order_type == self.SELL:
                    if ask < ticket.take_profit:
                        # 売りチケットを利確
                        profit = (ticket.open_price - ticket.take_profit) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_sell += profit
                    elif ask > ticket.stop_loss:
                        # 売りチケットを損切り
                        profit = (ticket.open_price - ticket.stop_loss) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_sell += profit

        elif action == self.BUY:
            ticket = Ticket(self.BUY, ask, ask + self.take_profit_pips * self.point,
                            ask - self.stop_loss_pips, self.lots)
            self.tickets.append(ticket)

        elif action == self.SELL:
            ticket = Ticket(self.SELL, bid, bid - self.take_profit_pips * self.point,
                            bid + self.stop_loss_pips * self.point, self.lots)
            self.tickets.append(ticket)

        elif action == self.CLOSE:
            for ticket in self.tickets:
                if ticket.order_type == self.BUY:
                    profit = (bid - ticket.open_price) * ticket.lots
                    self.info.balance += profit
                    self.info.total_pips_buy += profit
                elif ticket.order_type == self.SELL:
                    profit = (ticket.open_price - ask) * ticket.lots
                    self.info.balance += profit
                    self.info.total_pips_sell += profit

        # インデックスinc
        self.read_index += 1

        # obs, reward, done, infoを返す
        obs = self.make_obs('ohlc_array')
        reward = self.info.total_pips_buy + self.info.total_pips_sell
        done = self.info.balance <= 0 or self.read_index >= len(self.data)
        info = self.info
        return obs, reward, done, info


    def _render(self, mode='human', close=False):
        return self.make_obs(mode)

    def make_obs(self, mode):
        """
        observation 作成
        """
        # 本ステップ対象のobsを取得 TODO
        target = self.data.iloc[:self.read_index][self.data.columns]

        # TODO assert mode != 'human', 'mode==humanは未実装'
        if mode == 'human':
            pass

        elif mode == 'ohlc_array':
            m1 = numpy.array(target.iloc[-1 * self.visible_bar:][target.columns])
            m5 = numpy.array(target.resample('5min').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])
            m15 = numpy.array(target.resample('15min').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])
            m30 = numpy.array(target.resample('30min').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])
            h1 = numpy.array(target.resample('1H').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])

            return numpy.array([m1, m5, m15, m30, h1])

from pandas.core import resample
def ohlc2(self):
    return self.agg({'open': 'first',
                     'high': 'max',
                     'low': 'min',
                     'close': 'last'})
pandas.core.resample.DatetimeIndexResampler.ohlc2 = ohlc2


def str_date_to_datetime_01(str_date: str) -> datetime:
    """
    yyyy-mm-ddTHH:MM:SS.ffffffZ -> datetime
    :param str_date:
    :return:
    """
    return datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%fZ')



class AccountInformation(object):

    def __init__(self, initial_balance):
        # 口座資金(含み益含む)
        self.balance = initial_balance
        # 口座資金
        self.fixed_balance = initial_balance
        # 総獲得pips(買い)
        self.total_pips_buy = 0
        # 総獲得pips(売り)
        self.total_pips_sell = 0

    def items(self):

        return [('balance', self.balance), ('fixed_balance', self.fixed_balance), ('total_pips_buy', self.total_pips_buy), ('total_pips_sell', self.total_pips_sell)]


class Ticket(object):
    """
    チケット
    """

    def __init__(self, order_type, open_price, take_profit, stop_loss, lots):
        # タイプ
        self.order_type = order_type
        # 約定価格
        self.open_price = open_price
        # 利食い価格
        self.take_profit = take_profit
        # 損切り価格
        self.stop_loss = stop_loss
        # ロット
        self.lots = lots