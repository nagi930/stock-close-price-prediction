class Preprocess:

    def __init__(self, df):
        self.df = df

    def add_indicator(self, indicator, *args):
        ind = indicator(self.df, *args)
        result = ind.compute()
        name = indicator.__name__ + '_' + '_'.join([str(arg) for arg in args])
        self.df[name] = result

    def dropna(self, inplace=False):
        if inplace:
            self.df.dropna(inplace=inplace)
        else:
            return self.df.dropna()

    def get_dataframe_size(self):
        return self.df.shape

    def get_dataframe_n_rows_from_last(self, n):
        return self.df.iloc[-n:, :]
