import pandas as pd
import numpy as np

class RiskMeasurements:
    def __init__(self, df):
        self.df = df.copy()
        self.df.dropna(inplace=True)

        if self.df.index.name == 'date':
            self.df.reset_index(inplace=True)

        self.df = self.df.sort_values(by='date')
        self.df['close'] = pd.to_numeric(self.df['close'], errors='coerce')
        self.df.dropna(subset=['close'], inplace=True)
        self.__log_returns()

    def __log_returns(self):
        self.df['Log_Returns'] = np.log(self.df['close'] / self.df['close'].shift(1))

    def log_returns(self):
        return self.df[['date', 'Log_Returns']].dropna()

    def historical_volatility(self):
        log_ret = self.df['Log_Returns'].dropna()
        std_log_ret = log_ret.std()
        return {
            'vol_per_year': std_log_ret * np.sqrt(252),
            'vol_per_month': std_log_ret * np.sqrt(21),
            'vol_per_week': std_log_ret * np.sqrt(5),
            'vol_per_day': std_log_ret
        }

    def parametric_var(self, z=1.96, confidence_level=0.95):
        vols = self.historical_volatility()
        last_price = self.df['close'].iloc[-1]
        return {
            'value_at_risk': -z * vols['vol_per_day'] * last_price,
            'value_at_risk_annualized': -z * vols['vol_per_year'] * last_price,
            'value_at_risk_monthly': -z * vols['vol_per_month'] * last_price,
            'value_at_risk_weekly': -z * vols['vol_per_week'] * last_price,
            'confidence_level': confidence_level
        }

    def sharpe_ratio(self, risk_free_rate=0.06):
        log_ret = self.df['Log_Returns'].dropna()
        mean_ret = log_ret.mean() * 252
        vol = self.historical_volatility()['vol_per_year']
        return {
            'sharpe_ratio': (mean_ret - risk_free_rate) / vol,
            'risk_free_rate': risk_free_rate
        }

    def max_drawdown(self):
        log_ret = self.df['Log_Returns'].dropna()
        cumulative = np.exp(log_ret.cumsum())
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': self.df['date'].iloc[drawdown.idxmin()]
        }

    def kurtosis(self):
        return {
            'kurtosis': self.df['Log_Returns'].dropna().kurtosis()
        }

    def full_process(self):
        return {
            'historical_volatility': self.historical_volatility(),
            'parametric_var': self.parametric_var(),
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'kurtosis': self.kurtosis()
        }
