from asyncio.base_futures import _future_repr_info
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

import os

TRADE_START_TIME = 34200
TRADE_END_TIME = 57600
TICK_SIZE = 0.5

BID = "bid"
ASK = "ask"
DIRECTIONS = (BID, ASK)
MIN_STEP = 100


def get_daily_metadata(dataset_path):
    message_per_day = {}
    orderbook_per_day = {}
    for filename in tqdm(os.listdir(dataset_path)):
        day = filename.split("_")[1]
        if "message" in filename:
            data = pd.read_csv(dataset_path + filename, header=None)
            message_per_day[day] = data
        else:
            assert "orderbook" in filename
            data = pd.read_csv(dataset_path + filename, header=None)
            orderbook_per_day[day] = data

    return message_per_day, orderbook_per_day


def get_l2_snapshots(message_per_day, orderbook_per_day, day):
    message = message_per_day[day]
    orderbook = orderbook_per_day[day]
    times = message[0]

    message_index = 0
    book_indices = []
    for time in np.arange(TRADE_START_TIME + TICK_SIZE, TRADE_END_TIME - TICK_SIZE, TICK_SIZE):
        while message_index < len(times) and time > times[message_index]:
            message_index += 1
        book_indices.append(message_index - 1)
        if message_index == len(times):
            print("surpassed max message time")
            break

    return orderbook.iloc[book_indices], book_indices


class OrderBook:
    def __init__(self, order_book_snapshot):
        price_indices = {ASK: np.arange(10) * 4, BID: np.arange(10) * 4 + 2}
        amount_indices = {ASK: np.arange(10) * 4 + 1, BID: np.arange(10) * 4 + 3}
        self.price = {dir: order_book_snapshot[price_indices[dir]] for dir in DIRECTIONS}
        self.amount = {dir: order_book_snapshot[amount_indices[dir]] for dir in DIRECTIONS}

    def best_price(self, dir):
        return self.price[dir][0]

    def best_amount(self, dir):
        return self.amount[dir][0]

    def mid_price(self):
        return (self.best_price(ASK) + self.best_price(BID)) / 2

    def __str__(self):
        str = "price  / amount\n"
        for i in range(10):
            str += f"{self.price[ASK][9 - i]} / {self.amount[ASK][i]}\n"
        str += "----------\n"
        for i in range(10):
            str += f"{self.price[BID][i]} / {self.amount[BID][i]}\n"
        return str


def get_order_books(l2_snapshots):
    return [OrderBook(order_book_snapshot) for order_book_snapshot in l2_snapshots.values]


class OrderExecution:
    dir_index = 5
    price_index = 4
    amount_index = 3
    event_index = 1
    time_index = 0

    def __init__(self, message_snapshot):
        assert OrderExecution.is_execution_snapshot(message_snapshot)
        if message_snapshot[self.dir_index] == -1:
            self.dir = ASK
        else:
            self.dir = BID
        self.price = message_snapshot[self.price_index]
        self.amount = message_snapshot[self.amount_index]
        self.time = message_snapshot[self.time_index]

    @staticmethod
    def is_execution_snapshot(message_snapshot):
        return message_snapshot[OrderExecution.event_index] == 4

    def __str__(self):
        return f"price: {self.price}\namount: {self.amount}\ntime: {self.time}\ndir: {self.dir}"


def get_trades_features(messages, indices):
    total_trade_amount = {dir: [] for dir in DIRECTIONS}
    total_turnover = {dir: [] for dir in DIRECTIONS}
    current_trade_amount = {dir: 0 for dir in DIRECTIONS}
    current_turnover = {dir: 0 for dir in DIRECTIONS}

    for message in messages.values:
        if OrderExecution.is_execution_snapshot(message):
            order_execution = OrderExecution(message)
            current_trade_amount[order_execution.dir] += order_execution.amount
            current_turnover[order_execution.dir] += order_execution.price * order_execution.amount

        for dir in DIRECTIONS:
            total_trade_amount[dir].append(current_trade_amount[dir])
            total_turnover[dir].append(current_turnover[dir])

    prev_trade_amount = {dir: 0 for dir in DIRECTIONS}
    prev_turnover = {dir: 0 for dir in DIRECTIONS}
    tick_trade_amounts = {dir: [] for dir in DIRECTIONS}
    tick_turnovers = {dir: [] for dir in DIRECTIONS}

    for index in indices:
        for dir in DIRECTIONS:
            trade_amount = total_trade_amount[dir][index] - prev_trade_amount[dir]
            turnover = total_turnover[dir][index] - prev_turnover[dir]
            tick_trade_amounts[dir].append(trade_amount)
            tick_turnovers[dir].append(turnover)
            prev_trade_amount[dir] = total_trade_amount[dir][index]
            prev_turnover[dir] = total_turnover[dir][index]

    for dir in DIRECTIONS:
        assert len(tick_trade_amounts[dir]) == len(indices)
        assert len(tick_turnovers[dir]) == len(indices)

    return tick_trade_amounts, tick_turnovers


class Feature:
    def __init__(self, order_book: OrderBook, prev_order_book: OrderBook, tick_trade_amount, tick_turnover):
        # volumes on each price level
        # spread
        # best_price diff with last tick
        # time diff ???
        # dir_traded_amount
        # dir_turnover
        self.order_book = order_book
        self.prev_order_book = prev_order_book
        self.book_features = self._calculate_book_features()
        self.tick_trade_amount = tick_trade_amount
        self.tick_turnover = tick_turnover
        self.tick_features = self._calculate_tick_features()
        self.features = np.concatenate([self.book_features, self.tick_features])
        assert self.features.shape == (27,)

    def _calculate_book_features(self):
        spread = [self.order_book.best_price(ASK) - self.order_book.best_price(BID)]
        ask_volumes = self.order_book.amount[ASK]
        bid_volumes = self.order_book.amount[BID]
        ask_price_diff = [self.order_book.best_price(ASK) - self.prev_order_book.best_price(ASK)]
        bid_price_diff = [self.order_book.best_price(BID) - self.prev_order_book.best_price(BID)]

        return np.concatenate([spread, ask_volumes, bid_volumes, ask_price_diff, bid_price_diff])

    def _calculate_tick_features(self):
        features = []
        for dir in DIRECTIONS:
            features.append(self.tick_trade_amount[dir])
            features.append(self.tick_turnover[dir])
        return features

    def get_features(self):
        return self.features


def get_features(books, trade_amounts, turnovers):
    features = []
    for i in range(len(books) - 1):
        prev_book = books[i]
        book = books[i + 1]
        tick_amount = {dir: trade_amounts[dir][i + 1] for dir in DIRECTIONS}
        tick_turnover = {dir: turnovers[dir][i + 1] for dir in DIRECTIONS}
        feature = Feature(prev_book, book, tick_amount, tick_turnover).get_features()
        features.append(feature)
    assert len(features) + 1 == len(books)
    return np.array(features)


def get_mid_price_target(books, window_size):
    books = books[1:]  # features starting from first order_book
    targets = []
    for i in range(len(books) - window_size):
        current_book = books[i]
        current_mid_price = current_book.mid_price()
        future_mid_prices = []
        for j in range(window_size):
            future_book = books[i + j + 1]
            future_mid_prices.append(future_book.mid_price())

        mean_future_price = np.mean(future_mid_prices)
        price_diff = mean_future_price - current_mid_price
        targets.append(price_diff)
    assert len(targets) == len(books) - window_size
    return np.array(targets)


def get_features_and_targets(message_per_day, orderbook_per_day):
    features_per_day = {}
    targets_per_day = {}
    for day in tqdm(message_per_day):
        l2_snapshots, indices = get_l2_snapshots(message_per_day, orderbook_per_day, day)
        books = get_order_books(l2_snapshots)
        trade_amounts, turnovers = get_trades_features(message_per_day[day], indices)
        features = get_features(books, trade_amounts, turnovers)
        targets = get_mid_price_target(books, 80)
        features_per_day[day] = features
        targets_per_day[day] = targets

    return features_per_day, targets_per_day
