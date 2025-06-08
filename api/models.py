from django.db import models


#COLUNAS DE MARKET DATA: Date	Close	High	Low	Open	Volume	Symbol
class MarketData(models.Model):
    date = models.DateField()
    close = models.DecimalField(max_digits=20, decimal_places=2)
    high = models.DecimalField(max_digits=20, decimal_places=2)
    low = models.DecimalField(max_digits=20, decimal_places=2)
    open = models.DecimalField(max_digits=20, decimal_places=2)
    volume = models.BigIntegerField()
    symbol = models.CharField(max_length=10)
    class Meta:
        db_table = 'market_data'
        unique_together = ('date', 'symbol')
        ordering = ['date']
        verbose_name = 'Market Data'
        verbose_name_plural = 'Market Data Records'
    def __str__(self):
        return f"{self.symbol} - {self.date} - Close: {self.close}"