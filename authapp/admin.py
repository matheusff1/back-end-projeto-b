from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User

@admin.register(User)
class UsuarioAdmin(UserAdmin):
    model = User
    list_display = ('id', 'email', 'username', 'is_active', 'is_staff')
    ordering = ('id',)
    search_fields = ('email', 'username')