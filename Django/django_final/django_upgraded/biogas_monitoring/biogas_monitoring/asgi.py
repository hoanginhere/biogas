"""
ASGI config for biogas_monitoring project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os
from channels.auth import AuthMiddlewareStack
from channels.routing import URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
# from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter
from django.core.asgi import get_asgi_application
import datamanagement.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'biogas_monitoring.settings')

application = ProtocolTypeRouter(
    {
        'http':get_asgi_application(),
        "websocket":AllowedHostsOriginValidator(AuthMiddlewareStack(URLRouter(datamanagement.routing.websocket_urlpatterns))),
    }
)

