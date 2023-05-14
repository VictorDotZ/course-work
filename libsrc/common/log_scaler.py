import numpy as np


class LogScaler:
    def __init__(self, df) -> None:
        self.x_0 = None
        self.df = df

    def fit_transform(self, df):
        self.x_0 = df.iloc[0]
        return self.__get_log_returns(df)

    def transform(self, df):
        return self.__get_log_returns(df)

    def inverse_transform(self, values):
        pass

    def __get_log_returns(self, df):
        return np.log(df) - np.log(df.shift(1))
