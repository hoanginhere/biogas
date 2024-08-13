from django.db.models.signals import post_save
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from usermanagement.models import Warnings


@receiver(post_save, sender=Warnings)
def notification_created(sender, instance, created, **kwargs):
    if created:
        print("this activated")
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            'public_room',
            {
                "type": "send_notification",
                "message": instance.WarningContent
            }
        )
