from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Indicator(ABC):

    def __init__(self, df):
        self.close = df['Close']

    @abstractmethod
    def compute(self):
        pass


class Rsi(Indicator):

    def __init__(self, df, period):
        super().__init__(df)
        self.period = period

    def compute(self):
        print('RsiIndicator computed..')
        up = np.where(self.close.diff(1) > 0, self.close.diff(1), 0)
        down = np.where(self.close.diff(1) < 0, -1 * self.close.diff(1), 0)

        au = pd.Series(up).rolling(self.period).mean()
        ad = pd.Series(down).rolling(self.period).mean()
        rsi = np.array(au / (au + ad) * 100)
        return rsi


class Stochastic(Indicator):

    def __init__(self, df, n, m, slow=False):
        super().__init__(df)
        self.low = df['Low']
        self.high = df['High']
        self.n = n
        self.m = m
        self.slow = slow

    def compute(self):
        print('StochasticIndicator computed..')
        molecule = self.close - self.low.rolling(self.n).min()
        denominator = self.high.rolling(self.n).max() - self.low.rolling(self.n).min()
        stochastic = molecule / denominator * 100
        if self.slow:
            return stochastic.rolling(self.m).mean()
        return stochastic


class MeanAverage(Indicator):

    def __init__(self, df, n):
        super().__init__(df)
        self.n = n

    def compute(self):
        print('MeanAverageIndicator computed..')
        return self.close.rolling(self.n, min_periods=self.n).mean()


class Estrangement(Indicator):

    def __init__(self, df, n):
        super().__init__(df)
        self.n = n

    def compute(self):
        print('EstrangementIndicator computed..')
        return self.close / self.close.rolling(self.n, min_periods=self.n).mean() * 100
