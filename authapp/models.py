from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    # Senha é herdada de AbstractUser
    email = models.EmailField(unique=True)

    name = models.CharField(max_length=150, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    USERNAME_FIELD = 'email'         
    REQUIRED_FIELDS = ['username']    

    def __str__(self):
        return self.email