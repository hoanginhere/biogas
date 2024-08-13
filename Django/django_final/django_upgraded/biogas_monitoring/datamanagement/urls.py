from django.urls import path
from .views import *

urlpatterns = [
    path('add_machine/',add_machine),
    path('industrial/',industrial_gui),
    path('industrial/<str:mid>/',industrial_gui_1),
    path('loadgraphs/',loadgraph),
    path('intervals/',interval),
    path('controller/',controller_view),
    path('controller/<str:machine>/<str:status_sp>',controller_view_monitor),
    path('warnings/',warning_view),
    path('vibration/',vibration_view),
    path('vibration/result/<str:machine>',vibration_result),
    path('threshold',threshold),
]