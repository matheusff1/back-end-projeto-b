import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import clone_model
from api.models import MarketData, Prediction  



models_cfg = {
    'cp_tanh': {
        'model': 'lstm',
        'window_size': 30,
        'steps_out': 1,
        'pred_seq': 7,
        'epochs': 60,
        'batch_size': 64,
        'scaler_type': 'minmax',
        'train_ratio': 0.9,
        'activation': 'tanh',
        'callbacks': [
            EarlyStopping(monitor='val_mae', baseline=0.05,patience=20, restore_best_weights=True)
        ]
    },
    'af_relu': {
        'model': 'lstm',
        'window_size': 90,
        'steps_out': 7,
        'pred_seq': 7,
        'epochs': 60,
        'batch_size': 32,
        'scaler_type': 'minmax',
        'train_ratio': 0.9,
        'activation': 'relu',
        'callbacks': [
            EarlyStopping(monitor='val_mae', baseline=0.05,patience=20, restore_best_weights=True)
        ]
    },
}

BASE_CFG = models_cfg['cp_tanh']
BASE_WINDOW_SIZES = [BASE_CFG['window_size']]

model_cp_tanh = Sequential([
    LSTM(160, input_shape=(models_cfg['cp_tanh']['window_size'], 1), activation=models_cfg['cp_tanh']['activation'], return_sequences=False),
    Dense(1)
])

model_af_relu = Sequential([
    LSTM(160, input_shape=(models_cfg['af_relu']['window_size'], 1), activation=models_cfg['af_relu']['activation'], return_sequences=False),
    Dense(models_cfg['af_relu']['steps_out'])
])


class PredictionModel:
    def __init__(self, model, callbacks, df, window_size, epochs, batch_size, steps_out, pred_seq, multi_output=False, train_ratio=0.9, scaler_type='minmax'):
        self.df = df
        self.scaler_type = scaler_type
        self.model = clone_model(model)
        self.scaler = scaler_type
        self.window_size = window_size
        self.steps_out = steps_out
        self.pred_seq = pred_seq
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.multi_output = multi_output
        self.results = {}
       


    def create_shaped_df(self):
        X, y = [], []

        for i in range(self.window_size, len(self.df) - self.steps_out + 1):
            X.append(self.df.iloc[i - self.window_size:i, :].values)

            y.append(self.df['close'].iloc[i:i + self.steps_out].values)

        return np.array(X), np.array(y)
    
    def split_data(self, X, y):
        train_size = int(len(X) * self.train_ratio)
        
        train_X = X[:train_size]
        train_y = y[:train_size]
        test_X = X[train_size:]
        test_y = y[train_size:]

        print(f'Shapes de train_X: {train_X.shape}, train_y: {train_y.shape}')
        print(f'Shapes de test_X: {test_X.shape}, test_y: {test_y.shape}')

        return train_X, test_X, train_y, test_y

    def scale_data(self, train_X, test_X, train_y, test_y):
        
        if self.scaler_type == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
        else:
            raise ValueError("scaler_type deve ser 'standard' ou 'minmax'.")

        # escalar X
        shape_X = train_X.shape
        train_X_reshaped = train_X.reshape(-1, shape_X[-1])
        test_X_reshaped = test_X.reshape(-1, shape_X[-1])

        train_X_scaled = scaler_X.fit_transform(train_X_reshaped).reshape(shape_X)
        test_X_scaled = scaler_X.transform(test_X_reshaped).reshape(test_X.shape)

        # escalar y
        if self.multi_output:
            # y tem múltiplas saídas 
            scaler_y.fit(train_y)
            train_y_scaled = scaler_y.transform(train_y)
            test_y_scaled = scaler_y.transform(test_y)
        else:
            # y univariado
            train_y = train_y.reshape(-1, 1)
            test_y = test_y.reshape(-1, 1)

            scaler_y.fit(train_y)
            train_y_scaled = scaler_y.transform(train_y).flatten()
            test_y_scaled = scaler_y.transform(test_y).flatten()

        return train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled, scaler_X, scaler_y
    
    def process_data(self):
        X, y = self.create_shaped_df()
        train_X, test_X, train_y, test_y = self.split_data(X, y)
        train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled, scaler_X, scaler_y = self.scale_data(train_X, test_X, train_y, test_y)

        return train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled, scaler_X, scaler_y
    
    def next_days_prediction(self, last_data):
        predictions = []
        current_data = last_data.copy()

        
        for _ in range(self.pred_seq):
            pred = self.model.predict(current_data)
            predictions.append(pred)  
            current_data = np.roll(current_data, -1)  
            current_data[-1] = pred  
            
        return np.array(predictions)

    def fit_model(self, train_X, train_y, test_X, test_y):
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        result = self.model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=self.callbacks)
        self.results = result

    def predict(self, last_data, scaler_y):
        last_data = last_data.reshape(1, self.window_size, -1)

        print(f'Last data shape: {last_data.shape}')

        if(self.multi_output):
            predictions = self.model.predict(last_data)
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
        else:
            predictions = self.next_days_prediction(last_data)
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
        
        print(f'Predictions: {predictions.shape}')

        return predictions
    
    def run(self):
        train_X, test_X, train_y, test_y, scaler_X, scaler_y = self.process_data()
        self.fit_model(train_X, train_y, test_X, test_y)

        last_data = train_X[-1]
        predictions = self.predict(last_data, scaler_y)

        return {
            'predictions': predictions
        }
    

def run_all_predictions(df, model_cfg=models_cfg['cp_tanh'], model=model_cp_tanh):
    predictions = []
    df = df.copy()

    for symbol in df['symbol'].unique():
        print(f'Processing symbol: {symbol}')

        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        symbol_df = symbol_df.sort_values(by='date')
        symbol_df = symbol_df.drop(columns=['symbol', 'date', 'volume', 'high', 'low','open','id'], errors='ignore')
        print(f'Symbol DataFrame shape: {symbol_df.shape}, columns: {symbol_df.columns.tolist()}')
        
        model_runner = PredictionModel(
            model=model,
            callbacks=model_cfg['callbacks'],
            df=symbol_df,
            window_size=model_cfg['window_size'],
            steps_out=model_cfg['steps_out'],
            pred_seq=model_cfg['pred_seq'],
            epochs=model_cfg['epochs'],
            batch_size=model_cfg['batch_size'],
            train_ratio=model_cfg['train_ratio'],
            scaler_type=model_cfg['scaler_type']
        )
        
        result = model_runner.run()

        save = {
            'symbol': symbol,
            'predictions': result['predictions'],
            'results': model_runner.results.history,
            'date': pd.Timestamp.now()
        }

        predictions.append(save)
    
    return predictions

def predictions_process():
    df = pd.DataFrame(list(MarketData.objects.all().values()))

    results = run_all_predictions(df, model_cfg=models_cfg['cp_tanh'], model=model_cp_tanh)
    
    for result in results:  
        prediction = Prediction(
            date=pd.Timestamp.now(),
            results=result['results'],
            symbol=result['symbol'],
            prediction=result['predictions'].tolist()
        )
        prediction.save()