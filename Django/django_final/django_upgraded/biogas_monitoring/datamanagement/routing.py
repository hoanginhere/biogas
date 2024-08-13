from django.urls import re_path
from .consumers import *
from django.urls import path

websocket_urlpatterns = [
    re_path(r'ws/app/common',ChatConsumer.as_asgi()),
    re_path(r'ws/app/(?P<mid>\w+)',Chatchat.as_asgi()),
    path('ws/noti',warnings.as_asgi()),
    path('ws/notinum',notinum.as_asgi()),
    path('ws/moderator',ModDataConsumer.as_asgi()),
    path('ws/control_status',ControlStatus.as_asgi()),
    path('ws/vibration_result/<str:mid>',VibrationResult.as_asgi()),
    path('ws/interval/',IntervalConsumer.as_asgi())
]