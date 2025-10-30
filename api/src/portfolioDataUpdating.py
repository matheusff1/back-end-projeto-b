from decimal import Decimal
import pandas as pd
from django.utils import timezone
from api.models import Portfolio, PortfolioTracking, MarketData 

def update_portfolio_tracking(portfolio_id):
    try:
        portfolio = Portfolio.objects.get(id=portfolio_id)
    except Portfolio.DoesNotExist:
        print(f"Portfolio with id {portfolio_id} does not exist.")
        return False
    
    portfolio_last_data = PortfolioTracking.objects.filter(
        portfolio_id=portfolio_id
    ).order_by('-date').first()
    
    if not portfolio_last_data:
        print(f"No previous tracking found for portfolio {portfolio_id}. Skipping.")
        return False
    
    portfolio_last_distribution = portfolio_last_data.distribution
    
    if not portfolio_last_distribution:
        print(f"No distribution data found for portfolio {portfolio_id}.")
        return False

    latest_market_data = MarketData.objects.filter(
        symbol__in=portfolio_last_distribution.keys()
    ).order_by('-date').first()
    
    if not latest_market_data:
        print(f"No market data available for portfolio {portfolio_id} assets.")
        return False
    
    latest_market_date = latest_market_data.date
    if isinstance(latest_market_date, timezone.datetime):
        latest_market_date = latest_market_date.date()
    
    last_tracking_date = portfolio_last_data.date
    if isinstance(last_tracking_date, timezone.datetime):
        last_tracking_date = last_tracking_date.date()
    
    if latest_market_date <= last_tracking_date:
        print(f"Portfolio {portfolio_id}: No new market data. Last: {last_tracking_date}, Latest: {latest_market_date}. Skipping.")
        return False
    
    print(f"Portfolio {portfolio_id}: New market data! Last: {last_tracking_date}, New: {latest_market_date}")
    
    
    portfolio_last_balance = Decimal(portfolio_last_data.balance)
    portfolio_quantities = []
    
    for asset, alloc in portfolio_last_distribution.items():
        asset_data = MarketData.objects.filter(
            symbol=asset,
            date=last_tracking_date
        ).first()
        
        if not asset_data:
            asset_data = MarketData.objects.filter(
                symbol=asset,
                date__lte=last_tracking_date
            ).order_by('-date').first()
        
        if asset_data:
            price = Decimal(str(asset_data.close))
            alloc = Decimal(str(alloc))
            
            if price <= 0:
                print(f"  Invalid price for {asset}: {price}")
                continue
            
            quantity = (alloc * portfolio_last_balance) / price
            portfolio_quantities.append((asset, quantity))
            print(f"  {asset}: {quantity:.4f} shares @ ${price}")
        else:
            print(f"  No market data for {asset} on or before {last_tracking_date}")
            continue
    
    if not portfolio_quantities:
        print(f"No valid quantities calculated for portfolio {portfolio_id}.")
        return False
    
    
    portfolio_current_balance = Decimal('0.0')
    assets_current_values = {}
    
    for asset, quantity in portfolio_quantities:
        asset_data = MarketData.objects.filter(
            symbol=asset,
            date=latest_market_date
        ).first()
        
        if not asset_data:
            asset_data = MarketData.objects.filter(
                symbol=asset
            ).order_by('-date').first()
        
        if asset_data:
            price = Decimal(str(asset_data.close))
            
            if price <= 0:
                print(f"  Invalid current price for {asset}: {price}")
                continue
                
            value = quantity * price
            portfolio_current_balance += value
            assets_current_values[asset] = value
            print(f"  {asset}: ${value:,.2f} ({quantity:.4f} × ${price})")
        else:
            print(f"  CRITICAL: No current market data found for {asset}")
            return False
    
    
    portfolio_current_distribution = {}
    for asset, value in assets_current_values.items():
        if portfolio_current_balance > 0:
            portfolio_current_distribution[asset] = float(value / portfolio_current_balance)
        else:
            portfolio_current_distribution[asset] = 0.0
    

    market_datetime = timezone.datetime.combine(
        latest_market_date,
        timezone.datetime.min.time()
    ).replace(tzinfo=timezone.get_current_timezone())
    
    tracking, created = PortfolioTracking.objects.update_or_create(
        portfolio_id=portfolio_id,
        date=market_datetime,
        defaults={
            'balance': portfolio_current_balance,
            'distribution': portfolio_current_distribution
        }
    )
    
    
    action = "created" if created else "updated"
    balance_change = portfolio_current_balance - portfolio_last_balance
    balance_change_pct = (balance_change / portfolio_last_balance * 100) if portfolio_last_balance > 0 else 0
    
    print(f"\nPortfolio {portfolio_id} {action}!")
    print(f"   Market date: {latest_market_date}")
    print(f"   Executed at: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Old balance: ${portfolio_last_balance:,.2f}")
    print(f"   New balance: ${portfolio_current_balance:,.2f}")
    print(f"   Change: ${balance_change:,.2f} ({balance_change_pct:+.2f}%)\n")
    
    return True


def update_all_portfolios_tracking():
    portfolios = Portfolio.objects.all()
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    print(f"\n{'='*70}")
    print(f"Starting portfolio tracking update")
    print(f"Execution time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    for portfolio in portfolios:
        print(f"\n--- Processing Portfolio {portfolio.id}: {portfolio.name} ---")
        try:
            result = update_portfolio_tracking(portfolio.id)
            if result:
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error updating portfolio {portfolio.id}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"Portfolio tracking update completed")
    print(f"{'='*70}")
    print(f"  Updated: {success_count}")
    print(f"  Skipped (no new data): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total portfolios: {portfolios.count()}")
    print(f"{'='*70}\n")
    
