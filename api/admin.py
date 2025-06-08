from django.contrib import admin

from .models import MarketData
@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'symbol', 'close', 'high', 'low', 'open', 'volume')
    search_fields = ('symbol',)
    list_filter = ('date', 'symbol')
    ordering = ('-date',)
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
