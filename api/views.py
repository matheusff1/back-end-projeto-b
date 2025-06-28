from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from django.http import JsonResponse
from .models import MarketData, Prediction
from django.views.decorators.http import require_http_methods
from .src.RiskMeasurements import *
from .src.chat_bot_connection import get_chat_analysis
import json
import traceback

def get_sharpe_ratio(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        data = request.GET.get('data')
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close')))
            df.set_index('date', inplace=True)

            risk_free_rate = float(data['risk_free_rate']) 

            calc = RiskMeasurements(df)
            sharpe_ratio = calc.sharpe_ratio(risk_free_rate=risk_free_rate)
            return JsonResponse({'symbol': symbol, 'sharpe_ratio': sharpe_ratio}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def get_volatility(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close')))
            df.set_index('date', inplace=True)

            calc = RiskMeasurements(df)
            volatility = calc.historical_volatility()
            return JsonResponse({'symbol': symbol, 'volatility': volatility}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def get_var(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        data = request.GET.get('data')
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close')))
            df.set_index('date', inplace=True)
            z = float(data['z']) 
            confidence_level = float(data['confidence_level'])
            calc = RiskMeasurements(df)
            var = calc.parametric_var(z=z, confidence_level=confidence_level)
            return JsonResponse({'symbol': symbol, 'value_at_risk': var}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def get_kurtosis(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close')))
            df.set_index('date', inplace=True)

            calc = RiskMeasurements(df)
            kurtosis = calc.kurtosis()
            return JsonResponse({'symbol': symbol, 'kurtosis': kurtosis}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)


def get_max_drawdown(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close')))
            df.set_index('date', inplace=True)

            calc = RiskMeasurements(df)
            max_drawdown = calc.max_drawdown()
            return JsonResponse({'symbol': symbol, 'max_drawdown': max_drawdown}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def get_full_data(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close','high','low','open','volume')))
            df.set_index('date', inplace=True)

            calc = RiskMeasurements(df)
            full_data = calc.full_process()

            import json
            cleaned_data = json.loads(json.dumps(full_data, default=str))
            return JsonResponse({'symbol': symbol, 'full_data': cleaned_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


def get_chatbot_analysis(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        prompt = request.GET.get('prompt', """
            Você é um assistente financeiro que analisa dados de mercado e retorna insights resumidos e objetivos.
            Você irá receber dados de mercado e medidas estatísticas, como volatilidade, risco, outros indicadores financeiros e previsões.
            Você deve gerar uma análise detalhada e objetiva.
            
            Por favor, gere um relatório detalhado com:
            - Principais riscos
            - Recomendações
            - Notícias relacionadas ao ativo
            - Interpretação dos dados""")
        
        try:
            analysis = get_chat_analysis(prompt=prompt, symbol=symbol)
            return JsonResponse({'symbol': symbol, 'analysis': analysis}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)


def get_all_symbols(request):
    if request.method == 'GET':
        try:
            symbols = MarketData.objects.order_by('symbol').values_list('symbol', flat=True).distinct()
            return JsonResponse({'symbols': list(symbols)}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)


def get_historical_data(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            market_data = MarketData.objects.filter(symbol=symbol).order_by('date')
            if not market_data.exists():
                return JsonResponse({'error': 'No market data found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(market_data.values('date', 'close')))

            historical_data = df.to_dict(orient='records')
            return JsonResponse({'symbol': symbol, 'historical_data': historical_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@csrf_exempt
def get_optimized_portfolio(request):
    if request.method == 'POST':
        data = request.body
        try:
            data = json.loads(data)
            symbols = data.get('symbols',[])
            optimizer = data.get('optimizer')
            behaviour = data.get('behaviour')
            min_return = data.get('min_return')

            print(f'Received data: symbols={symbols}, optimizer={optimizer}, behaviour={behaviour}, min_return={min_return}')

            symbols_data = MarketData.objects.filter(symbol__in=symbols).order_by('date')
            symbols_data = pd.DataFrame(list(symbols_data.values('symbol', 'date', 'close', 'high', 'low', 'open', 'volume')))


            if(optimizer == 'markowitz'):
                data = process_markowitz_data(symbols_data, behaviour, min_return)
                optimization = PortfolioOptimizer(items=data['items'],items_val=data['items_val'], items_returns= data['items_returns'], items_pred= data['items_pred'],
                                                  items_vol= data['items_vol'], min_return=data['min_return'], optimizer= data['optimizer'], behaviour= data['behaviour'])
                results = optimization.optimize()

            elif(optimizer in ['gnosse','gnosse2']):
                predictions = Prediction.objects.filter(symbol__in=symbols).order_by('date')
                predictions_data = pd.DataFrame(list(predictions.values('date', 'results', 'symbol','prediction')))
                predictions_data = predictions_data.drop_duplicates(subset=['symbol'], keep='last')
                data = process_gnosse_data(symbols_data, predictions_data, optimizer, behaviour)

                optimization = PortfolioOptimizer(items=data['items'],items_val=data['items_val'], items_returns= data['items_returns'], items_pred= data['items_pred'],
                                                  items_vol= data['items_vol'], min_return=data['min_return'], optimizer= data['optimizer'], behaviour= data['behaviour'])
                results = optimization.optimize()

            return JsonResponse({'results': results}, status=200)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

def get_starter_portfolio(request):
    if request.method == 'GET':
        try:
            data = MarketData.objects.all().order_by('date')
            data = pd.DataFrame(list(data.values('symbol', 'date', 'close')))

            processed_data = process_markowitz_data(data, behaviour='neutral', min_return=0.0003)

            optimizer = PortfolioOptimizer(items=processed_data['items'], items_val=processed_data['items_val'],
                                           items_returns=processed_data['items_returns'], items_pred=processed_data['items_pred'],
                                           items_vol=processed_data['items_vol'], min_return=processed_data['min_return'],
                                           optimizer=processed_data['optimizer'], behaviour=processed_data['behaviour'])
            
            results = optimizer.optimize()
            to_json_results = json.loads(json.dumps(results, default=str))
            return JsonResponse({'symbols': to_json_results['items'], 'distribuition': to_json_results['optimized_weights']  , 'complete_result': to_json_results}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@csrf_exempt
def get_portfolio_full_analysis(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            symbols = data.get('symbols', [])
            distribution = data.get('distribution', [])

            distribution = [float(d) for d in distribution]

            interval = 3 

            symbols_data = MarketData.objects.filter(symbol__in=symbols).order_by('date').filter(date__gte=pd.Timestamp.now() - pd.DateOffset(years=interval))
            symbols_data = pd.DataFrame(list(symbols_data.values('symbol', 'date', 'close', 'high', 'low', 'open', 'volume')))

            symbols_prices = symbols_data.drop_duplicates(subset=['symbol'], keep='last').sort_values(by='symbol')
            symbols_prices = symbols_prices.set_index('symbol')['close'].apply(float).to_dict()


            portfolio_risk = PortfolioRisk(symbols=symbols, distribution=distribution, price_dict=symbols_prices, df=symbols_data)
            results = portfolio_risk.full_process()

            return JsonResponse({'symbols': symbols, 'measures': results, 'status': 200})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_asset_predictions(request, symb):
    if request.method == 'GET':
        symbol = request.GET.get('symbol', symb)
        try:
            predictions = Prediction.objects.filter(symbol=symbol).order_by('date')
            if not predictions.exists():
                return JsonResponse({'error': 'No predictions found for the given symbol.'}, status=404)

            df = pd.DataFrame(list(predictions.values('date', 'results', 'symbol', 'prediction')))
            df = df.drop_duplicates(subset=['symbol'], keep='last')

            prediction_data = json.loads(df.to_json(orient='records'))
            return JsonResponse({'symbol': symbol, 'predictions': prediction_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method.'}, status=405)