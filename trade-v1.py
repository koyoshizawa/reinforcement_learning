# -*- coding:utf-8 -*-

# TODO ある１時点のRSI or 何かINDEXを学習データとして、その後5期間 らへんの相場観(ボラティリティ?)を分類する
#

"""
1. oanda api からデータを取得し、dbに保存
2. データを食わせられる形に整形
3. actionの決定
4. トレードの実行
"""

import sqlite3
import oandapy
from datetime import datetime, time

# TODO 本番は別ファイルで管理
API_KEY = 'c0ea162478beff0aaafb9275eb125153-f26e4988e7cd9236cd52e4320d163f70'
ACCOUNT_ID = '7559147'
DB_NAME = '/Users/KoYoshizawa/PycharmProjects/reinforcement_learning/fx_trade.db'


def str_date_to_datetime_01(str_date: str) -> datetime:
    """
    yyyy-mm-ddTHH:MM:SS.ffffffZ -> datetime
    :param str_date:
    :return:
    """
    return datetime.strptime(str_date, '%Y-%m-%dT%H:%M:%S.%fZ')

def datetime_to_unixtime(date_time):
    """
    datetime -> unixtime
    :param date_time:
    :return:
    """
    return int(time.mktime(date_time.timetuple()))

def save_data():

    # データ取得
    oanda = oandapy.API(access_token=API_KEY, environment='practice')
    response = oanda.get_prices(instruments='USD_JPY')['prices']
    instrument = response[0]['instrument']
    bid = float(response[0]['bid'])
    ask = float(response[0]['ask'])
    time = response[0]['time']

    # DBに保存
    # 前提
    # c.execute('''CREATE TABLE fx_data (id integer primary key autoincrement, time text, instrument text, bid, ask)''')
    # は実行済み

    # 接続
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    sql = "INSERT INTO fx_data(time, instrument, bid, ask) VALUES (?, ?, ?, ?);"
    values = (time, instrument, bid, ask)
    c.execute(sql, values)

    conn.commit()
    conn.close()


def edit_data():

    # DBからデータ取得
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    sql = """SELECT *
             FROM fx_data 
             WHERE id =(select max(id) from fx_data);"""
    c.execute(sql)

    data = dict()
    for row in c:
        data = {
            'id': row[0],
            'time': row[1],
            'bid': row[3],
            'ask': row[4],
        }

    conn.close()
    return data

class AccountInformation(object):

    def __init__(self):

        # # 初回のみ
        # initial_balance = 1000000
        # # 口座資金(含み益含む)
        # self.balance = initial_balance
        # # 口座資金
        # self.fixed_balance = initial_balance
        # # 総獲得pips(買い)
        # self.total_pips_buy = 0
        # # 総獲得pips(売り)
        # self.total_pips_sell = 0
        # conn = sqlite3.connect(DB_NAME)
        # c = conn.cursor()
        # sql = """CREATE TABLE account_information (id integer primary key autoincrement, balance, fixed_balance, total_pips_buy, total_pips_sell)"""
        # c.execute(sql)

        # DBから口座情報取得取得
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = """SELECT *
                 FROM account_information
                 WHERE id =(select max(id) from account_information);"""
        c.execute(sql)

        for row in c:
            self.balance = row[1]
            self.fixed_balance = row[2]
            self.total_pips_buy = row[3]
            self.total_pips_sell = row[4]
        conn.close()

    def items(self):

        return [('balance', self.balance), ('fixed_balance', self.fixed_balance), ('total_pips_buy', self.total_pips_buy), ('total_pips_sell', self.total_pips_sell)]

    def save(self):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = """INSERT INTO account_information
                 (balance, fixed_balance, total_pips_buy, total_pips_sell)
                 VALUES (?, ?, ?, ?);"""
        values = (self.balance, self.fixed_balance, self.total_pips_buy, self.total_pips_sell)
        c.execute(sql, values)
        conn.commit()
        conn.close()



class Agent(object):
    """
    ロジックに基づいてアルゴリズムを選択し、取引を行うためのエージェントクラス
    """

    def __init__(self, bid, ask):

        self.ask = ask
        self.bid = bid

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
        self.take_profit_pips = 0
        self.loss_cut_pips = 0
        self.current_position = ''

        self.chart_monitor = ChartMonitor(bid, ask)
        self.info = AccountInformation()
        self.oanda = oandapy.API(access_token=API_KEY, environment='practice')
        tickets = self.oanda.get_trades(ACCOUNT_ID)['trades']
        self.tickets = tickets[0] if len(tickets) !=0 else {'units':0}


        # 前回取引情報を取得
        self.get_prev_data()


    def save(self):
        """
        DB (status, position)
        """
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = """INSERT INTO agent(status, position, created_at) 
                 VALUES (?, ?, ?);"""
        values = (self.current_status, self.current_position, datetime.now())
        c.execute(sql, values)
        conn.commit()
        conn.close()

    def get_prev_data(self):

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = "SELECT * FROM agent WHERE id =(select max(id) from agent);"
        c.execute(sql)

        for row in c:
            self.id = row[0]
            self.current_status = row[1]
            self.current_position = row[2]

        conn.close()


    def exec_trade(self):
        """
        現在の状態に合わせて利確or損切りを実施し、次の状態に遷移させる。
        :return: 取引があった場合はTrue ない場合はFalse
        """
        is_trade = False
        win_lose = ''

        # 初回取引の場合に実行
        if self.current_position == '':
            self._first_trade()

        # 現在の状態から利確or損切りを行うべきか判断し、実行する
        elif self.current_status == self.A:  # A
            is_trade, win_lose = self._check_status_a()
            if is_trade:
                # ポジション解消
                self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])

        elif self.current_status == self.B:  # B
            is_trade, win_lose = self._check_status_b()
            if is_trade:
                # ポジション解消
                self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])

        elif self.current_status == self.C:  # C
            is_trade, win_lose = self._check_status_c()
            if is_trade:
                # ポジション解消
                self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])

        elif self.current_status == self.D:  # D
            is_trade, win_lose = self._check_status_d()
            if is_trade:
                # ポジション解消
                self.oanda.close_trade(ACCOUNT_ID, self.tickets['id'])

        # ポジションの解消が行われた場合、現在枚数、ポジションレートを初期化し、次の取引ロジックを実行
        if is_trade:
            # 取引の勝ち負けによって次実行するルール確定
            next_status = self._get_acceptable_method_type()[win_lose]

            # 次の取引ルールを実行
            eval('self._change_status_to_{}'.format(next_status))(win_lose)

            # 状態C, Dの取引チェックに使用する情報を初期化
            self.chart_monitor.init_over_top_below_bottom()

            # 情報を保存
            self.save()

    def _first_trade(self):
        """
        1回目のトレード用 A or Bを実行する
        """
        if (self.ask + self.bid) / 2 - self.chart_monitor.prev_top > self.take_profit_pips:
            self._change_status_to_a('start')
            self.save()
        elif (self.ask + self.bid) / 2  - self.chart_monitor.prev_bottom < -1 * self.take_profit_pips:
            self._change_status_to_b('start')
            self.save()

    def _check_status_a(self):
        """
        状態Aの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win', 'lose', 'keep'
        """
        if self.ask - self.chart_monitor.prev_bottom < -1 * self.take_profit_pips:
            return True, 'win'
        elif self.ask - self.tickets['price'] > self.loss_cut_pips:
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
        elif self.bid - self.tickets['price'] < -1 * self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def _check_status_c(self):
        """
        状態Cの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win' 'lose', 'keep'
        """

        # 直近の山を上回ってから、直近の谷を指定の値以上下回る
        if self.chart_monitor.has_over_top and self.bid - self.chart_monitor.prev_bottom < -1 * self.loss_cut_pips:
            return True, 'win'
        elif self.bid - self.tickets['price'] < -1 * self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def _check_status_d(self):
        """
        状態Dの時に実行。利確or損切りをするかどうかを判断する。
        :return is_trade bool: ポジション解消を行うべきかの真偽を返す
        :return win_lose str: ポジション解消による勝ち負けを返す 'win' 'lose', 'keep'
        """
        # 直近の谷を下回ってから、直近の山を指定の値以上上回る
        if self.chart_monitor.has_bellow_bottom and self.ask - self.chart_monitor.prev_top > self.loss_cut_pips:
            return True, 'win'
        elif self.ask - self.tickets['price'] > self.loss_cut_pips:
            return True, 'lose'
        return False, 'keep'

    def _change_status_to_a(self, win_lose):
        """
        状態Aに遷移し、ポジションを持つ
        """
        # 初回はunits -> 0
        if win_lose == 'start':
            self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY',
                                    units=self.UNITS_TABLE[0][win_lose] * self.base_lots,
                                    side='sell', type='market')
        else:
            self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY',
                                    units=self.UNITS_TABLE[self.tickets['units']/self.base_lots][win_lose] * self.base_lots,
                                    side='sell', type='market')
        self.current_position = self.SHORT
        self.current_status = self.A

    def _change_status_to_b(self, win_lose):
        """
        状態Bに遷移し、ポジションを持つ
        """
        # 初回はunits->0
        if win_lose == 'start':
            self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY',
                                    units=self.UNITS_TABLE[0][win_lose] * self.base_lots,
                                    side='buy', type='market')
        else:
            self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY',
                                    units=self.UNITS_TABLE[self.tickets['units']/self.base_lots][win_lose] * self.base_lots,
                                    side='buy', type='market')
        self.current_position = self.LONG
        self.current_status = self.B

    def _change_status_to_c(self, win_lose):
        """
        状態Cに遷移し、ポジションを持つ
        """
        self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY',
                                units=self.UNITS_TABLE[self.tickets['units']/self.base_lots][win_lose] * self.base_lots,
                                side='buy', type='market')
        self.current_position = self.LONG
        self.current_status = self.C

    def _change_status_to_d(self, win_lose):
        """
        状態Dに遷移し、ポジションを持つ
        """
        self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY',
                                units=self.UNITS_TABLE[self.tickets['units']/self.base_lots][win_lose] * self.base_lots,
                                side='sell', type='market')
        self.current_position = self.SHORT
        self.current_status = self.D


    def _get_acceptable_method_type(self) -> dict:
        """
        現在の状態から選択可能なロジックを確定させる
        :return: {'win': 勝った時の実行ロジック, 'lose': 負けた時の実行ロジック}
        """
        d = {'win': '', 'lose': ''}

        if self.current_status == self.A:  # A
            d['win'] = self.B  # B
            d['lose'] = self.C  # C

        elif self.current_status == self.B:  # B
            d['win'] = self.A  # A
            d['lose'] = self.D  # D

        elif self.current_status == self.C:  # C
            d['win'] = self.A  # A
            d['lose'] = self.D  # D

        elif self.current_status == self.D:  # D
            d['win'] = self.B  # B
            d['lose'] = self.C  # C

        return d


class ChartMonitor(object):
    """
    model (id, trend, prev_mid, prev_top, prev_bottom, has_bellow_bottom, has_over_top, created_at)
    """
    def __init__(self, bid, ask):
        self.datetime = datetime.now()  # データの日時
        self.bid = bid
        self.ask = ask
        self.mid = (ask + bid) / 2

        self.standard_rate = 0

        self.id = 0
        self.trend = 0  # -1 下降トレンド 1 上昇トレンド
        self.prev_mid = 0  # 1つ前のレート
        self.prev_top = 0  # 直前の山
        self.prev_bottom = 0  # 直前の谷
        self.has_bellow_bottom = False  # 直前の谷を下回った -> True (取引を行うたびにFalseに初期化する必要あり)
        self.has_over_top = False  # 直前の山を上回った -> True (取引を行うたびにFalseに初期化する必要あり

        # trend prev_mid prev_top prev_bottom has_bellow_bottom has_over_top を前回のデータで更新 (2回目以降)
        self.get_prev_data()

        # trend prev_mid prev_top prev_bottom has_bellow_bottom has_over_top を今回用に更新
        self.update()

        # 今回用に更新されたデータを保存
        self.save()


    def get_prev_data(self):

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = "SELECT * FROM chart_monitor WHERE id =(select max(id) from chart_monitor);"
        c.execute(sql)

        for row in c:
            self.id = row[0] + 1
            self.trend = row[1]
            self.prev_mid = row[2]
            self.prev_top = row[3]
            self.prev_bottom = row[4]
            self.has_bellow_bottom = row[5]
            self.has_over_top = row[6]

        conn.close()


    def update(self) -> bool:

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
        elif self.mid> self.prev_top:
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

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = """UPDATE chart_monitor SET has_bellow_bottom = ? WHERE id = ?;"""
        values = (self.has_bellow_bottom, self.id)
        c.execute(sql, values)
        sql = """UPDATE chart_monitor SET has_over_top = ? WHERE id = ?;"""
        values = (self.has_over_top, self.id)
        c.execute(sql, values)
        conn.commit()
        conn.close()

    def save(self):

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = """INSERT INTO chart_monitor(trend, prev_mid, prev_top, prev_bottom, has_bellow_bottom, has_over_top, created_at) 
                 VALUES (?, ?, ?, ?, ?, ?, ?);"""
        values = (self.trend, self.prev_mid, self.prev_top, self.prev_bottom, self.has_bellow_bottom, self.has_over_top, datetime.now())
        c.execute(sql, values)
        conn.commit()
        conn.close()

def is_trade_time(now):

    # 月:0 -> 日:6
    start_time = time(9, 0, 0, 0)
    end_time = time(23, 0, 0, 0)

    # 火 -> 木 は無条件に実行
    if 1 <= now.weekday() <= 3:
        return True

    # 月はstart_time以降 金はend_time以前のみ実行
    if (now.weekday() == 0 and start_time <= now.time()) or (now.weekday() == 4 and end_time >= now.time()):
        return True

    return False

def close_all_trade():

    oanda = oandapy.API(access_token=API_KEY, environment='practice')
    tickets = oanda.get_trades(account_id=ACCOUNT_ID)['trades']
    for ticket in tickets:
        oanda.close_order(account_id=ACCOUNT_ID, order_id=ticket['id'])

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    sql = "DELETE FROM chart_monitor;"
    c.execute(sql)
    sql = "DELETE FROM agent;"
    c.execute(sql)
    conn.close()

def main():

    # 実行時間かチェック
    now = datetime.now()
    if is_trade_time(now):
        # 金曜の早めの時間に残っているチケットを全て決済して終了する(DB更新忘れずに)

        save_data()
        data = edit_data()

        agent = Agent(data['bid'], data['ask'])
        agent.exec_trade()

    # 実行時間を超えている場合は残っているチケットを全て決済して終了する
    # チケットがなければpass
    else:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        sql = "SELECT COUNT(*) FROM agent;"
        c.execute(sql)

        for d in c:
            if d[0] > 0:
                close_all_trade()
            else:
                pass


if __name__ == '__main__':
    main()




