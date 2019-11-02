# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import pandas as pd

from abc import abstractmethod
from gym.spaces import Space, Box
from typing import List, Dict

from tensortrade.trades import Trade, TradeType
from tensortrade.exchanges import InstrumentExchange
from tensortrade.slippage import RandomUniformSlippageModel
from tensortrade.features import FeaturePipeline


class SimulatedExchange(InstrumentExchange):
    """An instrument exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.

    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(base_instrument=kwargs.get('base_instrument', 'USD'),
                         dtype=kwargs.get('dtype', np.float16),
                         feature_pipeline=kwargs.get('feature_pipeline', None))

        self._commission_percent = kwargs.get('commission_percent', 0.3)
        self._base_precision = kwargs.get('base_precision', 2)
        self._instrument_precision = kwargs.get('instrument_precision', 8)
        self._min_trade_price = kwargs.get('min_trade_price', 1E-6)
        self._max_trade_price = kwargs.get('max_trade_price', 1E6)
        self._min_trade_amount = kwargs.get('min_trade_amount', 1E-3)
        self._max_trade_amount = kwargs.get('max_trade_amount', 1E6)
        self._min_order_amount = kwargs.get('min_order_amount', 1E-3)

        self._initial_balance = kwargs.get('initial_balance', 1E4)
        self._observation_columns = kwargs.get(
            'observation_columns', ['open', 'high', 'low', 'close', 'volume'])
        self._price_column = kwargs.get('price_column', 'close')
        self._window_size = kwargs.get('window_size', 1)
        self._pretransform = kwargs.get('pretransform', True)
        self._price_history = None

        self.data_frame = data_frame

        self._max_price_slippage_percent = kwargs.get('max_allowed_slippage_percent', 1.0)
        self._max_amount_slippage_percent = kwargs.get('max_allowed_slippage_percent', 0.0)

        SlippageModelClass = kwargs.get('slippage_model', RandomUniformSlippageModel)
        self._price_slip_model = SlippageModelClass(min=self._min_trade_price,
                                                    max=self._max_trade_price,
                                                    slip=self._max_price_slippage_percent)
        self._amount_slip_model = SlippageModelClass(min=self._min_trade_amount,
                                                    max=self._max_trade_amount,
                                                    slip=self._max_amount_slippage_percent)

    @property
    def data_frame(self) -> pd.DataFrame:
        """The underlying data model backing the price and volume simulation."""
        return getattr(self, '_data_frame', None)

    @data_frame.setter
    def data_frame(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self._data_frame = data_frame
            return

        self._data_frame = data_frame[self._observation_columns]
        self._price_history = data_frame[self._price_column]

        if self._pretransform:
            self.transform_data_frame()

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline=FeaturePipeline):
        self._feature_pipeline = feature_pipeline

        if isinstance(self.data_frame, pd.DataFrame) and self._pretransform:
            self.transform_data_frame()

        return self._feature_pipeline

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def portfolio(self) -> Dict[str, float]:
        return self._portfolio

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades

    @property
    def performance(self) -> pd.DataFrame:
        return self._performance

    @property
    def generated_space(self) -> Space:
        low = np.array([self._min_trade_price, ] * (len(self._observation_columns)-1) + [self._min_trade_amount, ])
        high = np.array([self._max_trade_price, ] * (len(self._observation_columns)-1) + [self._max_trade_amount, ])

        return Box(low=low, high=high, dtype='float')

    @property
    def generated_columns(self) -> List[str]:
        return list(self._observation_columns)

    @property
    def has_next_observation(self) -> bool:
        return self._current_step < len(self._data_frame) - 1

    def _next_observation(self) -> pd.DataFrame:
        lower_range = max((self._current_step - self._window_size, 0))
        upper_range = max(min(self._current_step, len(self._data_frame)), 1)

        obs = self._data_frame.iloc[lower_range:upper_range]

        if len(obs) < self._window_size:
            padding = np.zeros((len(self.generated_columns), self._window_size - len(obs)))
            obs = pd.concat([pd.DataFrame(padding), obs], ignore_index=True)

        if not self._pretransform and self._feature_pipeline is not None:
            obs = self._feature_pipeline.transform(obs, self.generated_space)

        self._current_step += 1

        return obs

    def transform_data_frame(self) -> bool:
        if self._feature_pipeline is not None:
            self._data_frame = self._feature_pipeline.transform(self._data_frame,
                                                                self.generated_space)

    def current_price(self, symbol: str) -> float:
        if self._price_history is not None:
            return float(self._price_history.iloc[self._current_step])
        return 0

    def _is_valid_trade(self, trade: Trade) -> bool:
        # print('_is_valid_trade')
        if trade.valid:
            return True

        if trade.is_hold:
            return trade.transact_amount <= self.portfolio.get(trade.symbol, 0)
        elif trade.is_buy:
            return trade.transact_amount >= self._min_order_amount and self.balance >= trade.transact_total
        elif trade.is_sell:
            return trade.transact_amount >= self._min_order_amount and self.portfolio.get(trade.symbol, 0) >= trade.transact_amount

        return False

    def _update_account(self, trade: Trade):
        # print('update')
        log = {'step': self._current_step}
        if trade.is_buy:
            log['action'] = trade.log
            self._balance -= trade.transact_total
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) + trade.transact_amount
        elif trade.is_sell:
            log['action'] = trade.log
            self._balance += trade.transact_total
            self._portfolio[trade.symbol] = self._portfolio.get(trade.symbol, 0) - trade.transact_amount
        elif trade.is_hold:
            log['action'] = trade.log
        else:
            log['action'] = "Unknown Trade Type: {}".format(trade.to_dict)

        if self._is_valid_trade(trade) and not trade.is_hold:
            self._trades = self._trades.append({'step': self._current_step,
                                                'symbol': trade.symbol,
                                                'type': trade.trade_type,
                                                'amount': trade.transact_amount,
                                                'price': trade.transact_total}, ignore_index=True)

        self._portfolio[self._base_instrument] = self._balance

        log.update({'balance': self.balance,
                    'net_worth': self.net_worth
                    })
        # print(log['action'])
        # print(self.balance, self.net_worth)
        self._performance = self._performance.append( log , ignore_index=True)

    def execute_trade(self, trade: Trade) -> Trade:
        # print('execute')
        trade = trade.copy()
        transact_amount = trade.order_amount
        transact_price = trade.order_price

        if trade.is_hold:
            # hodl boyz!
            trade.transact_amount = transact_amount
            trade.transact_price = transact_price
        else:
            trade.order_commission = transact_price * self.commission_percent

        if trade.is_buy:
            if trade.is_limit_buy:
                # our buying power will fluxuate slightly as the market moves
                slip_adjusted_amount = (self.balance * (1-(self._max_amount_slippage_percent/100)) * transact_amount) / transact_price
                transact_amount = self._amount_slip_model.random_slip(slip_adjusted_amount)
            elif trade.is_market_buy:
                # the price can fluxuate up or down during the market buy order
                transact_price = self._price_slip_model.random_slip(transact_price)
            else:
                # catch all
                transact_price = self._bind_trade_price(transact_price)
                transact_amount = self._bind_trade_amount(transact_amount)

        elif trade.is_sell:
            if trade.is_limit_sell:
                # if we're selling we can only sell LESS than we have on hand.
                amount = self._amount_slip_model.slip_down(transact_amount)
            elif trade.is_market_sell:
                # sell price can fluxuate up or down during a market sale order
                transact_price = self._price_slip_model.random_slip(transact_price)
            else:
                # catch all
                transact_price = self._bind_trade_price(transact_price)
                transact_amount = self._bind_trade_amount(transact_amount)


        # save all the
        trade.transact_price = round(transact_price, self._base_precision)
        trade.transact_amount = round(transact_amount, self._instrument_precision)
        trade.transact_commission = round((transact_price*transact_amount) * (self.commission_percent), self._base_precision)
        trade.transact_commission_percent=self.commission_percent
        trade.transact_total = round((transact_price * transact_amount) * (1 + self.commission_percent), self._base_precision)

        if self._is_valid_trade(trade):
            trade.valid = True

            if not trade.is_hold:
                trade.executed = True

            # print('Valid',self.balance, trade.to_dict)
            self._update_account(trade)
        else:
            trade.valid = False
            trade.executed = False
            # print("Invalid trade", self.balance, trade.to_dict)
        return trade

    def reset(self):
        super().reset()

        self._current_step = 0
        self._balance = self.initial_balance
        self._portfolio = {self.base_instrument: self.balance}
        self._trades = pd.DataFrame([], columns=['step', 'symbol', 'type', 'amount', 'price'])
        self._performance = pd.DataFrame([], columns=['balance', 'net_worth', 'action', 'step'])
