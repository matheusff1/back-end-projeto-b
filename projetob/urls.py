"""
URL configuration for projetob project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from api.views import get_sharpe_ratio as sharpe_ratio_view
from api.views import get_volatility as volatility_view
from api.views import get_var as var_view
from api.views import get_max_drawdown as max_drawdown_view
from api.views import get_kurtosis as kurtosis_view
from api.views import get_chatbot_analysis as chatbot_analysis_view
from api.views import get_full_data as get_full_data
from api.views import get_all_symbols as get_all_symbols
from api.views import get_starter_portfolio as get_starter_portfolio
from api.views import get_historical_data as get_historical_data
from api.views import get_optimized_portfolio as get_optimized_portfolio
from api.views import get_portfolio_full_analysis as get_portfolio_full_analysis
from api.views import get_asset_predictions as get_asset_predictions


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/sharpe-ratio/<str:symb>/', sharpe_ratio_view, name='sharpe-ratio'),
    path('api/volatility/<str:symb>/', volatility_view, name='volatility'),
    path('api/var/<str:symb>/', var_view, name='var'),
    path('api/kurtosis/<str:symb>/', kurtosis_view, name='kurtosis'),
    path('api/chatbot-analysis/<str:symb>/', chatbot_analysis_view, name='chatbot-analysis'),
    path('api/full-data/<str:symb>/', get_full_data, name='full-data'),
    path('api/max-drawdown/<str:symb>/', max_drawdown_view, name='max-drawdown'),
    path('api/historical-data/<str:symb>/', get_historical_data, name='historical-data'),
    path('api/symbols/', get_all_symbols, name='get-all-symbols'),
    path('api/starter-portfolio/', get_starter_portfolio, name='get-starter-portfolio'), 
    path('api/optimized-portfolio/', get_optimized_portfolio, name='get-optimized-portfolio'),
    path('api/portfolio-analysis/', get_portfolio_full_analysis, name='get-portfolio-full-analysis'),
    path('api/asset-predictions/<str:symb>/', get_asset_predictions, name='get-asset-predictions'),
]