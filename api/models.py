from django.db import models
from django.conf import settings



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
    

class Prediction(models.Model):
    date = models.DateField()
    results = models.JSONField()
    symbol = models.CharField(max_length=10)
    prediction = models.JSONField(default=list)
    
    class Meta:
        db_table = 'predictions'
        unique_together = ('date', 'symbol')
        ordering = ['date']
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'
    
    def __str__(self):
        return f"{self.symbol} - {self.date} - Prediction: {self.prediction}"
    



class Portfolio(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='portfolios'
    )
    name = models.CharField(max_length=150)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    assets = models.JSONField(default=list)

    initial_distribution = models.JSONField(default=dict)

    current_distribution = models.JSONField(default=dict)

    initial_balance = models.DecimalField(max_digits=20, decimal_places=2, default=0)

    current_balance = models.DecimalField(max_digits=20, decimal_places=2, default=0)

    class Meta:
        db_table = 'portfolio'
        verbose_name = 'Portfolio'
        verbose_name_plural = 'Portfolios'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.user.email}"