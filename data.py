from FinanceDataReader import DataReader
from indicator import *


class Data:

    def __init__(self, stock_code: str, start: str, end: str):
        self.df = DataReader(stock_code, start, end)


class Preprocessor:

    def __init__(self, data: Data):
        self.data = data

    def add_indicator(self, indicator):
        ind = indicator(self.data)
        result = ind.compute()
        name = indicator.__name__.split('Indicator')[0]
        self.data.df[name] = result


def main():
    p = Preprocessor(Data('005930', '20210101', '20210131'))
    p.add_indicator(RsiIndicator)
    print(p.data.df)


if __name__ == '__main__':
    main()



