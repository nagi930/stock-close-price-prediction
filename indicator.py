from abc import ABC, abstractmethod


class Indicator(ABC):

    def __init__(self, data):
        self.close = data.df['Close']
        self.low = data.df['Low']
        self.high = data.df['Low']

    @abstractmethod
    def compute(self):
        pass


class RsiIndicator(Indicator):

    def __init__(self, data):
        super().__init__(data)

    def compute(self):
        print('RsiIndicator computed..')


class StochasticIndicator(Indicator):

    def __init__(self, data):
        super().__init__(data)

    def compute(self):
        print('StochasticIndicator computed..')


class MeanAverageIndicator(Indicator):

    def __init__(self, data):
        super().__init__(data)

    def compute(self):
        print('MeanAverageIndicator computed..')
