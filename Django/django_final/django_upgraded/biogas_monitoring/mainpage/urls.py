from django.urls import path
from .views import *

urlpatterns = [
    path('',landing),
    path('home/',homepage),
]