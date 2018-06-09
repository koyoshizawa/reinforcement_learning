# -*- coding:utf-8 -*-

"""
1. oanda api からデータを取得し、dbに保存
2. データを食わせられる形に整形
3. actionの決定
4. トレードの実行
"""

import sqlite3
import oandapy
from datetime import datetime
import time
import random

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


class Ticket(object):
    """
    チケット
    """
    def __init__(self, id, order_type, open_price, take_profit, stop_loss, lots):
        # id
        self.id = id
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

    # def open(self):
    #
    #     conn = sqlite3.connect(DB_NAME)
    #     c = conn.cursor()
    #     sql = """
    #           INSERT INTO tickets
    #           (order_type, open_price, take_profit, stop_loss, lots, is_active, oanda_id)
    #           VALUES(?, ?, ?, ?, ?, ?, ?);
    #           """
    #     values = (self.order_type, self.open_price, self.take_profit, self.stop_loss, self.lots, 1, self.oanda_id)
    #     c.execute(sql, values)
    #     conn.commit()
    #     conn.close()
    #
    # def close(self):
    #
    #     conn = sqlite3.connect(DB_NAME)
    #     c = conn.cursor()
    #     sql = """UPDATE tickets SET is_active = 0 WHERE oanda_id = ?;"""
    #     value = (self.oanda_id,)
    #     c.execute(sql, value)
    #     conn.commit()
    #     conn.close()


class Agent(object):

    def __init__(self, id, bid, ask):
        self.id = id
        self.bid = bid
        self.ask = ask

        self.HOLD = 0
        self.BUY = 1
        self.SELL = 2
        self.CLOSE = 3
        self.take_profit_pips = 30
        self.stop_loss_pips = 15
        self.point = 0.01
        self.lots = 1

        self.info = AccountInformation()
        self.oanda = oandapy.API(access_token=API_KEY, environment='practice')
        self.tickets = [ticket for ticket in self.get_active_tickets()]


    def get_active_tickets(self):


        tickets = self.oanda.get_trades(ACCOUNT_ID)['trades']
        # conn = sqlite3.connect(DB_NAME)
        # c = conn.cursor()
        # sql = """SELECT *
        #          FROM tickets
        #          WHERE is_active = 1;
        #       """
        # c.execute(sql)
        trades = []
        for ticket in tickets:
            if ticket['side'] == 'buy':
                t = Ticket(id=ticket['id'], order_type=ticket['side'], open_price=ticket['price'],
                           take_profit=ticket['price'] + self.take_profit_pips * self.point,
                           stop_loss=ticket['price'] - self.stop_loss_pips * self.point,
                           lots=ticket['units'])
            elif ticket['side'] == 'sell':
                t = Ticket(id=ticket['id'], order_type=ticket['side'], open_price=ticket['price'],
                           take_profit=ticket['price'] - self.take_profit_pips * self.point,
                           stop_loss=ticket['price'] + self.stop_loss_pips * self.point,
                           lots=ticket['units'])
            else:
                pass
            trades.append(t)

        return trades


    def exec_action(self, action):
        if action == self.HOLD:
            for ticket in self.tickets:
                if ticket.order_type == self.BUY:
                    if self.bid > ticket.take_profit:
                        # 買いチケットを利確
                        profit = (ticket.take_profit - ticket.open_price) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_buy += profit
                        self.oanda.close_trade(ACCOUNT_ID, ticket.id)
                    elif self.bid < ticket.stop_loss:
                        # 買いチケットを損切り
                        profit = (ticket.stop_loss - ticket.open_price) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_buy += profit
                        self.oanda.close_trade(ACCOUNT_ID, ticket.id)
                elif ticket.order_type == self.SELL:
                    if self.ask < ticket.take_profit:
                        # 売りチケットを利確
                        profit = (ticket.open_price - ticket.take_profit) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_sell += profit
                        self.oanda.close_trade(ACCOUNT_ID, ticket.id)
                    elif self.ask > ticket.stop_loss:
                        # 売りチケットを損切り
                        profit = (ticket.open_price - ticket.stop_loss) * ticket.lots
                        self.info.balance += profit
                        self.info.total_pips_sell += profit
                        self.oanda.close_trade(ACCOUNT_ID, ticket.id)

        elif action == self.BUY:
            response = self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY', units=self.lots, side='buy', type='market')

        elif action == self.SELL:
            response = self.oanda.create_order(ACCOUNT_ID, instrument='USD_JPY', units=self.lots, side='sell', type='market')

        elif action == self.CLOSE:
            for ticket in self.tickets:
                if ticket.order_type == self.BUY:
                    profit = (self.bid - ticket.open_price) * ticket.lots
                    self.info.balance += profit
                    self.info.total_pips_buy += profit
                elif ticket.order_type == self.SELL:
                    profit = (ticket.open_price - self.ask) * ticket.lots
                    self.info.balance += profit
                    self.info.total_pips_sell += profit
                self.oanda.close_trade(ACCOUNT_ID, ticket.id)

        # 口座情報保存
        self.info.save()


    pass

def main():
    save_data()
    data = edit_data()

    agent = Agent(data['id'], data['bid'], data['ask'])
    action = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 2])
    # action = random.choice([1])

    # action = 1
    agent.exec_action(action)

if __name__ == '__main__':
    main()