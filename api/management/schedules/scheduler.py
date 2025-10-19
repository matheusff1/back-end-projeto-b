from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from django.core.management import call_command
from pytz import timezone

def run_update_market_data_command():
    print(f"[{datetime.now()}] Executando comando update_market_data...")
    call_command('update_market_data') 

def run_predictions_command():
    print(f"[{datetime.now()}] Executando comando run_and_save_predictions...")
    call_command('run_and_save_predictions') 

def run_update_portfolios_data():
    print(f"[{datetime.now()}] Atualizando dados dos portfólios...")
    call_command('update_portfolios_data')

def start():
    scheduler = BackgroundScheduler(timezone=timezone('America/Sao_Paulo'))

    scheduler.add_job(
        run_update_market_data_command, 
        trigger=CronTrigger(hour=19, minute=0),
        misfire_grace_time=120 
    )

    scheduler.add_job(
        run_predictions_command, 
        trigger=CronTrigger(hour=20, minute=20),
        misfire_grace_time=60 
    )

    scheduler.add_job(
        run_update_portfolios_data,
        trigger=CronTrigger(hour=23, minute=30),
        misfire_grace_time=60
    )

    print(f"[{datetime.now()}] Scheduler iniciado...")
    scheduler.start()
