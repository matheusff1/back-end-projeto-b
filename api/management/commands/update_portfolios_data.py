from django.core.management.base import BaseCommand
from api.src.portfolioDataUpdating import *


class Command(BaseCommand):
    help = 'Atualiza os dados de portfólios diariamente'

    def handle(self, *args, **kwargs):
        print("Iniciando atualização diária dos portfólios...")
        update_all_portfolios_tracking()
        print("Atualização dos portfólios finalizada.")