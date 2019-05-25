from django.contrib import admin

from .models import Progress, History

# Register your models here.
admin.site.register([Progress, History])
