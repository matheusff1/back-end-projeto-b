import yfinance as yf
import pandas as pd
import requests
from decimal import Decimal
import os
import django
from django.conf import settings
from projetob.settings import TWELVE_DATA_API_KEY 

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "projetob.settings")
django.setup()

from api.models import MarketData  

API_KEY = TWELVE_DATA_API_KEY  

SYMBOLS_TD = ['PETR4', 'BRL/USD', 'BTC/USD', 'AAPL', 'XAU/USD', 'BVSP', 'GSPC', 'IXIC']
SYMBOLS_YF = [
    'PETR4.SA', 'BRL=X', 'BTC-USD', 'AAPL', '^BVSP', '^GSPC', '^IXIC', 'BZ=F', 'GC=F',
    'VALE3.SA', 'ITUB4.SA', 'B3SA3.SA', 'WEGE3.SA', 'BBAS3.SA', 'ABEV3.SA', 'RENT3.SA',
    'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JNJ', 'DIS',
    'BABA', 'TSM', 'SAP.DE', 'NESN.SW', '7203.T', 'RDSA.AS', 'BP.L', 'TM',
    'CL=F', 'NG=F', 'SI=F', 'ZC=F', 'ZS=F'
]


PERIOD_YF = 'max'
INTERVAL_YF = '1d'
INTERVAL_TD = '1day'
PERIOD_TD = 5000
URL = 'https://api.twelvedata.com/time_series'
TODAY = pd.Timestamp.now().normalize()


class DataCollector:
    def __init__(self, symbols_yf=SYMBOLS_YF, symbols_td=SYMBOLS_TD, url=URL, api_key=API_KEY,
                 today_date=TODAY, interval_yf=INTERVAL_YF, period_yf=PERIOD_YF,
                 interval_td=INTERVAL_TD, period_td=PERIOD_TD):
        self.symbols_yf = symbols_yf
        self.symbols_td = symbols_td
        self.api_key = api_key
        self.period_yf = period_yf
        self.interval_yf = interval_yf
        self.interval_td = interval_td
        self.period_td = period_td
        self.url = url
        self.today_date = today_date

    def collect_yfinance_data(self):
        for symbol in self.symbols_yf:
            try:
                data = yf.download(
                    tickers=symbol,
                    period=self.period_yf,
                    interval=self.interval_yf,
                    progress=True
                )

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

            
                if not data.empty:
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)


                    registros = []
                    for _, row in data.iterrows():
                        try:
                            registros.append(MarketData(
                                                    date=row['Date'].date(),
                                                    open=Decimal(str(row['Open'])),
                                                    high=Decimal(str(row['High'])),
                                                    low=Decimal(str(row['Low'])),
                                                    close=Decimal(str(row['Close'])),
                                                    volume=int(row['Volume']),
                                                    symbol=row['Symbol']
                                                ))
                        except Exception as e:
                            print(f"Erro ao preparar registro de {symbol} em {row['Date']}: {e}")

                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros salvos para {symbol}.")
                else:
                    print(f"Dados vazios para {symbol}.")
            except Exception as e:
                print(f"Erro ao coletar {symbol}: {e}")

    def collect_twelvedata_data(self):
        for symbol in self.symbols_td:
            print(f'Coletando dados de {symbol} via TwelveData...')

            params = {
                'symbol': symbol,
                'interval': self.interval_td,
                'outputsize': self.period_td,
                'apikey': self.api_key,
                'format': 'JSON'
            }

            response = requests.get(self.url, params=params)
            data = response.json()

            if data.get('status') == 'ok' and 'values' in data:
                df = pd.DataFrame(data['values'])
                df['symbol'] = symbol
                df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'datetime'], inplace=True)

                registros = []
                for _, row in df.iterrows():
                    try:
                        registros.append(MarketData(
                            date=pd.to_datetime(row['datetime']).date(),
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=int(float(row['volume'])),
                            symbol=row['symbol']
                        ))
                    except Exception as e:
                        print(f"Erro ao preparar registro de {symbol} em {row.get('datetime')}: {e}")

                MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                print(f"{len(registros)} registros salvos para {symbol}.")
            else:
                print(f"Erro ao buscar dados para {symbol}: {data.get('message', 'Sem mensagem de erro')}")

    def update_yfinance_data(self):
        today = pd.Timestamp(self.today_date)  
        for symbol in self.symbols_yf:
            print(f'Atualizando dados de {symbol} via yFinance...')

            try:
                ult = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
                if ult:
                    start_date = pd.Timestamp(ult.date) + pd.Timedelta(days=1)
                else:
                    start_date = pd.Timestamp('2000-01-01')

                end_date = today
                if start_date > end_date:
                    print(f"Dados de {symbol} já estão atualizados.")
                    continue

                data = yf.download(
                    tickers=symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=self.interval_yf,
                    progress=False
                )

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if not data.empty:
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

                    registros = []
                    for _, row in data.iterrows():
                        try:
                            registros.append(MarketData(
                                date=row['Date'].date(),
                                open=row['Open'],
                                high=row['High'],
                                low=row['Low'],
                                close=row['Close'],
                                volume=int(row['Volume']),
                                symbol=row['Symbol']
                            ))
                        except Exception as e:
                            print(f"Erro ao preparar registro: {e}")

                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros atualizados para {symbol}.")

                else:
                    print(f"Nenhum dado novo para {symbol}.")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")


    def update_twelvedata_data(self):
        today = self.today_date
        for symbol in self.symbols_td:
            print(f'Atualizando dados de {symbol} via TwelveData...')

            ult = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
            if ult:
                start_date = ult.date + pd.Timedelta(days=1)
            else:
                start_date = pd.to_datetime('2000-01-01')

            end_date = today
            if start_date > end_date:
                print(f"Dados de {symbol} já estão atualizados.")
                continue

            params = {
                'symbol': symbol,
                'interval': self.interval_td,
                'apikey': self.api_key,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'format': 'JSON',
                'timezone': 'UTC'
            }

            try:
                response = requests.get(self.url, params=params)
                data = response.json()

                if data.get('status') == 'ok' and 'values' in data:
                    df = pd.DataFrame(data['values'])
                    df['symbol'] = symbol
                    df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'datetime'], inplace=True)

                    registros = []
                    for _, row in df.iterrows():
                        try:
                            registros.append(MarketData(
                                date=pd.to_datetime(row['datetime']).date(),
                                open=row['open'],
                                high=row['high'],
                                low=row['low'],
                                close=row['close'],
                                volume=int(float(row['volume'])),
                                symbol=row['symbol']
                            ))
                        except Exception as e:
                            print(f"Erro ao preparar registro de {symbol} em {row.get('datetime')}: {e}")

                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros atualizados para {symbol}.")

                else:
                    msg = data.get('message', 'Sem mensagem de erro')
                    print(f"Erro ao buscar dados para {symbol}: {msg}")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")
