from api.models import PortfolioTracking, Portfolio, MarketData
import pandas as pd


def update_portfolio_tracking(portfolio_id):
    try:
        portfolio_last_data = PortfolioTracking.objects.filter(
            portfolio_id=portfolio_id
        ).order_by('-date').first()
    except PortfolioTracking.DoesNotExist:
        print(f"Portfolio with id {portfolio_id} does not exist.")
        return
    
    today = pd.Timestamp.now()
    portfolio_assets = portfolio_last_data.distribution.keys()
    portfolio_last_date = portfolio_last_data.date
    portfolio_last_balance = portfolio_last_data.balance
    portfolio_last_distribution = portfolio_last_data.distribution

    portfolio_quantities = []
    for asset, alloc in portfolio_last_distribution.items():
        asset_data = MarketData.objects.filter(
            symbol=asset,
        ).order_by('-date').first()
        if asset_data:
            price = asset_data.close
            quantity = (alloc) * portfolio_last_balance / price
            portfolio_quantities.append((asset, quantity))
        else:
            print(f"No market data for {asset} on {portfolio_last_date}")
            return
    
    portfolio_current_balance = 0.0
    assets_current_values = {}

    for asset, quantity in portfolio_quantities:
        asset_data = MarketData.objects.filter(
            symbol=asset,
        ).order_by('-date').first()
        if asset_data:
        
            price = asset_data.close
            value = quantity * price
            portfolio_current_balance += value
            assets_current_values[asset] = value
        else:
            print(f"No market data for {asset} on {today}")
            return
    
    
    portfolio_current_distribution = {}
    for asset in assets_current_values:
        portfolio_current_distribution[asset] = assets_current_values[asset] / portfolio_current_balance
    
    PortfolioTracking.objects.create(
        portfolio = portfolio_id,
        date = today,
        balance = portfolio_current_balance,
        distribution = portfolio_current_distribution
    )

    print(f"Portfolio {portfolio_id} tracking updated for date {today}.")


def update_all_portfolios_tracking():
    portfolios = Portfolio.objects.all()
    for portfolio in portfolios:
        try:
            update_portfolio_tracking(portfolio.id)
        except Exception as e:
            print(f"Error updating portfolio {portfolio.id}: {e}")
    print("All portfolios tracking update process completed.")