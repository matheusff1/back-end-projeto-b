from django.core.management.base import BaseCommand
from api.src.predictionModelling import *

class Command(BaseCommand):
    help = 'Executa as predições e salva os resultados no banco de dados'

    def handle(self, *args, **kwargs):
        print("Iniciando o processo de previsões.")
        try:
            #predictions_process()
            predictions_process_v2()
        except Exception as e:
            print(f"Ocorreu um erro ao executar o modelo de previsão: {e}")