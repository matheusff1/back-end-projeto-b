from apscheduler.schedulers.background import BackgroundScheduler
from django.core.management import call_command
from datetime import datetime


def run_update_market_data_command():
    print(f"[{datetime.now()}] Executando comando update_market_data...")
    call_command('update_market_data') 

def run_predictions_command():
    print(f"[{datetime.now()}] Executando comando run_and_save_predictions...")
    call_command('run_and_save_predictions') 

def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_update_market_data_command, 'cron', hour=19, minute=0)  
    scheduler.add_job(run_predictions_command, 'cron', hour=19, minute=20)
    scheduler.start()
