# a, b, c, d を判断
# input は1m 5m 15m 30m 1h のclose high low open

import numpy
import pandas
import gym

from datetime import datetime

from pandas.core import resample
def ohlc2(self):
    return self.agg({'open': 'first',
                     'high': 'max',
                     'low': 'min',
                     'close': 'last'})
pandas.core.resample.DatetimeIndexResampler.ohlc2 = ohlc2

class Ticket(object):
    """
    チケット
    """

    def __init__(self, order_type, open_price):
        # タイプ
        self.order_type = order_type
        # 約定価格
        self.open_price = open_price

class AccountInformation(object):
    def __init__(self):
        # 初回のみ
        initial_balance = 1000000
        # 口座資金(含み益含む)
        self.balance = initial_balance
        # 口座資金
        self.fixed_balance = initial_balance
        # 総獲得pips(買い)
        self.total_pips_buy = 0
        # 総獲得pips(売り)
        self.total_pips_sell = 0

class ChartMonitor(object):
    """
    model (id, trend, prev_mid, prev_top, prev_bottom, has_bellow_bottom, has_over_top, created_at)
    """

    def __init__(self):
        self.datetime = datetime.now()  # データの日時


        self.standard_rate = 0

        self.id = 0
        self.trend = 0  # -1 下降トレンド 1 上昇トレンド
        self.prev_mid = 0  # 1つ前のレート
        self.prev_top = 0  # 直前の山
        self.prev_bottom = 0  # 直前の谷
        self.has_bellow_bottom = False  # 直前の谷を下回った -> True (取引を行うたびにFalseに初期化する必要あり)
        self.has_over_top = False  # 直前の山を上回った -> True (取引を行うたびにFalseに初期化する必要あり

    def update(self, bid, ask) -> bool:

        self.bid = bid
        self.ask = ask
        self.mid = (ask + bid) / 2

        # 2回目のみ実行 1回目2回目のデータから現在のトレンドを取得する
        if self.prev_mid == 0:
            if self.mid > self.prev_mid:
                self.trend = 1
            else:
                self.trend = -1

        # 価格に指定した値以上の変動があるか
        if self._has_over_standard_rate():

            # 山or谷が発生しているか確認
            # self.current_trend -1 なのに 前回よりも価格が高い -> 前回の価格が谷
            if self.trend == -1 and self.prev_mid < self.mid:
                self.prev_bottom = self.prev_mid  # 谷の価格をセット
                self.trend = 1  # トレンドを上昇トレンドに変更
            # self.current_trend 1 なのに 前回よりも価格が低い -> 前回の価格が山
            elif self.trend == 1 and self.prev_mid > self.mid:
                self.prev_top = self.prev_mid  # 山の価格をセット
                self.trend = -1  # トレンドを下降トレンドに変更

            # 直近の谷を下回っているか、直近の谷を上回っているか判断
            self._check_over_top_below_bottom()

            # 次の判断の為に、現在の価格を前回の価格としてセット
            self.prev_mid = self.mid

            # 取引ロジックを実行する
            return True
        # 取引ロジックは実行しない
        self.prev_mid = self.mid
        return False


    def _has_over_standard_rate(self) -> bool:
        """
        指定した価格よりも変動が大きいかを判断する
        :return: 指定した価格よりも大きい->True
        """
        if abs(self.prev_mid - self.mid) > self.standard_rate:
            return True
        else:
            return False

    def _check_over_top_below_bottom(self):
        """
        直近の谷を下回っているか、直近の谷を上回っているか判断
        """
        if self.mid < self.prev_bottom:
            self.has_bellow_bottom = True
        elif self.mid > self.prev_top:
            self.has_over_top = True
        else:
            pass

    def init_over_top_below_bottom(self):
        """
        over_top below_bottom を初期化する
        状態C、Dの取引を実行する際に実行
        """
        self.has_over_top = False
        self.has_bellow_bottom = False

class MyEnv(gym.Env):

    def __init__(self):

        self.action_converter = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

        self.visible_bar = 5
        self.read_index = 200

        self.initial_balance = 1000000
        self.spread = 0.5
        self.point = 0.01
        self.action_space = gym.spaces.Discrete(4)
        self.data = self.get_data('fx_data.json')
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=numpy.shape(self.make_obs()))


    def get_data(self, file_path):
        # データ取得
        data = dict(pandas.read_json(file_path)['candles'])
        # DFに変形するために整形
        columns = ['time', 'open', 'high', 'low', 'close']
        # columns = ['time', 'close']

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
        df['time'] = df['time'].map(self.str_date_to_datetime_01)
        # timeをインデックスにセット
        df.set_index('time', inplace=True)
        return df

    def make_obs(self):
        """
        observation 作成
        """

        data = self.data

        # 本ステップ対象のobsを取得 TODO
        target = data.iloc[:self.read_index][data.columns]

        m1 = numpy.array(target.iloc[-1 * self.visible_bar:][target.columns])
        m5 = numpy.array(target.resample('5min').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])
        m15 = numpy.array(target.resample('15min').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])
        m30 = numpy.array(target.resample('30min').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])
        h1 = numpy.array(target.resample('1H').ohlc2().dropna().iloc[-1 * self.visible_bar:][target.columns])

        # 正規化(最後の1m closedataを基準とする)
        denom = m1[-1][0]
        m1 = m1 / denom
        m5 = m5 / denom
        m15 = m15 / denom
        m30 = m30 / denom
        h1 = h1 / denom

        return numpy.array([m1, m5, m15, m30, h1])

    def str_date_to_datetime_01(self, str_date: str) -> datetime:
        """
        yyyy-mm-ddTHH:MM:SS.ffffffZ -> datetime
        :param str_date:
        :return:
        """
        return datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%fZ')

    def _seed(self, seed=None):
        pass

    def _close(self):
        pass

    def _reset(self):
        self.info = AccountInformation()
        self.read_index = 200
        self.data = self.get_data('fx_data.json')
        self.agent = Agent()
        self.tickets = None

        return self.make_obs()

    def _render(self, mode='human', close=False):
        return self.make_obs()

    def _step(self, action):

        is_trade = False
        done = False
        init_obs = self.make_obs()
        current_data = self.data.iloc[self.read_index]
        ask = current_data['close'] + self.spread * self.point
        bid = current_data['close'] - self.spread * self.point
        self.agent.bid = bid
        self.agent.ask = ask
        self.agent.chart_monitor.update(bid, ask)

        action = self.action_converter[action]
        self.agent.open_trade(action)
        while True:

            if is_trade or done:
                break

            self.read_index += 1
            current_data = self.data.iloc[self.read_index]
            ask = current_data['close'] + self.spread * self.point
            bid = current_data['close'] - self.spread * self.point

            self.agent.bid = bid
            self.agent.ask = ask
            self.agent.chart_monitor.update(bid, ask)
            is_trade = self.agent.check_and_close_trade()
            # todo 変更の必要あり
            done = self.info.balance <= 0 or self.read_index > len(self.data) - 2

        next_obs = self.make_obs()
        reward = self.agent.info.total_pips_sell + self.agent.info.total_pips_buy
        # reward = 1 if self.agent.win_lose == 'win' else 0
        return next_obs, reward, done, {}


class Agent(object):


    def __init__(self):

        self.ask = 0
        self.bid = 0


        self.A = 'a'
        self.B = 'b'
        self.C = 'c'
        self.D = 'd'

        self.UNITS_TABLE = {
            0: {'start': 1},
            1: {'win': 1, 'lose': 9},
            3: {'win': 1, 'lose': 9},
            9: {'win': 3, 'lose': 12},
            12: {'win': 12, 'lose': 3},
        }

        self.LONG = 0
        self.SHORT = 1

        self.point = 0.01
        self.base_lots = 1
        self.take_profit_pips = 15 * self.point
        self.loss_cut_pips = 30 * self.point

        self.chart_monitor = ChartMonitor()
        self.info = AccountInformation()
        self.ticket = None

        self.current_status = ''

    def _check_status_a(self):
        """
        状態Aの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win', 'lose', 'keep'
        """
        if self.ask - self.chart_monitor.prev_bottom < -1 * self.take_profit_pips:
            return True, 'win'
        # elif self.ask - self.tickets['price'] > self.loss_cut_pips:
        elif self.ask - self.ticket.open_price > self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def _check_status_b(self):
        """
        状態Bの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win' 'lose', 'keep'
        """
        if self.bid - self.chart_monitor.prev_top > self.take_profit_pips:
            return True, 'win'
        # elif self.bid - self.tickets['price'] < -1 * self.loss_cut_pips:
        elif self.bid - self.ticket.open_price < -1 * self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def _check_status_c(self):
        """
        状態Cの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win' 'lose', 'keep'
        """

        # 直近の山を上回ってから、直近の谷を指定の値以上下回る
        if self.chart_monitor.has_over_top and self.bid - self.chart_monitor.prev_bottom < self.take_profit_pips * -1:
            return True, 'win'
        # elif self.bid - self.tickets['price'] < -1 * self.loss_cut_pips:
        elif self.bid - self.ticket.open_price < -1 * self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def _check_status_d(self):
        """
        状態Dの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win' 'lose', 'keep'
        """
        # 直近の谷を下回ってから、直近の山を指定の値以上上回る
        if self.chart_monitor.has_bellow_bottom and self.ask - self.chart_monitor.prev_top > self.take_profit_pips:
            return True, 'win'
        # elif self.ask - self.tickets['price'] > self.loss_cut_pips:
        elif self.ask - self.ticket.open_price > self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def check_and_close_trade(self):

        # 現在の状態から利確or損切りを行うべきか判断し、実行する
        is_trade = False
        if self.current_status == self.A:  # A
            is_trade, self.win_lose = self._check_status_a()
            if is_trade:
                # ポジション解消
                #                 self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])
                self.info.total_pips_sell = self.info.total_pips_sell + self.ticket.open_price - self.ask
                self.reward = self.info.total_pips_sell + self.ticket.open_price - self.ask
                self.ticket = None

        elif self.current_status == self.B:  # B
            is_trade, self.win_lose = self._check_status_b()
            if is_trade:
                # ポジション解消
                #                 self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])
                self.info.total_pips_buy = self.info.total_pips_buy + self.bid - self.ticket.open_price
                self.reward = self.info.total_pips_buy + self.bid - self.ticket.open_price
                self.ticket = None

        elif self.current_status == self.C:  # C
            is_trade, self.win_lose = self._check_status_c()
            if is_trade:
                # ポジション解消
                #                 self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])
                self.info.total_pips_buy = self.info.total_pips_buy + self.bid - self.ticket.open_price
                self.reward = self.info.total_pips_buy + self.bid - self.ticket.open_price
                self.ticket = None

        elif self.current_status == self.D:  # D
            is_trade, self.win_lose = self._check_status_d()
            if is_trade:
                # ポジション解消
                #                 self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])
                self.info.total_pips_sell = self.info.total_pips_sell + self.ticket.open_price - self.ask
                self.reward = self.info.total_pips_sell + self.ticket.open_price - self.ask
                self.ticket = None

        return is_trade

    def open_trade(self, action):
        eval('self._change_status_to_{}'.format(action))()
        self.chart_monitor.init_over_top_below_bottom()


    def _change_status_to_a(self):
        """
        状態Aに遷移し、ポジションを持つ
        """

        self.ticket = Ticket('sell', self.bid)
        self.current_position = self.SHORT
        self.current_status = self.A

    def _change_status_to_b(self):
        """
        状態Bに遷移し、ポジションを持つ
        """

        self.ticket = Ticket('buy', self.ask)
        self.current_position = self.LONG
        self.current_status = self.B

    def _change_status_to_c(self):
        """
        状態Cに遷移し、ポジションを持つ
        """

        self.ticket = Ticket('buy', self.ask)
        self.current_position = self.LONG
        self.current_status = self.C

    def _change_status_to_d(self):
        """
        状態Dに遷移し、ポジションを持つ
        """

        self.ticket = Ticket('sell', self.bid)
        self.current_position = self.SHORT
        self.current_status = self.D