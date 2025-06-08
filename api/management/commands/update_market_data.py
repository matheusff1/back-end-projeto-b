from django.core.management.base import BaseCommand
from api.src.loadData import DataCollector  

class Command(BaseCommand):
    help = 'Atualiza os dados de mercado diariamente'

    def handle(self, *args, **kwargs):
        print("Iniciando atualização diária...")
        collector = DataCollector()
        collector.update_yfinance_data()
        print("Atualização finalizada.")