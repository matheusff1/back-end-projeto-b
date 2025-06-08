from django.shortcuts import render
import pandas as pd
from django.http import JsonResponse
from .models import MarketData
from django.views.decorators.http import require_http_methods
from .src.RiskMeasurements import RiskMeasurements
from .src.chat_bot_connection import get_chat_analysis


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

            df = pd.DataFrame(list(market_data.values('date', 'close')))
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
            -Resumo geral e interpretação dos dados""")
        
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