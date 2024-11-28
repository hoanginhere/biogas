import os
import django
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ['DJANGO_SETTINGS_MODULE'] = 'biogas_monitoring.settings'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "biogas_monitoring.settings")
django.setup()
from usermanagement.models import Warnings
from datamanagement.models import Machine
import random
import datetime
# import json
import time

content_list = [
    "Current of g01 is over the limit",
    "Voltage phase A of g02 is over the limit",
    "Voltage phase B of g02 is over the limit",
    "Voltage phase C is over the limit",
]

type_list = [
    "ELE"
]

# Warnings.objects.all().delete()

for i in range(10):
    x = Warnings(WarningContent=(content_list[random.randint(0,3)]+' '+str(int(datetime.datetime.now().timestamp()))),WarningType="ELE",Machines=Machine.objects.get(id="1"))
    time.sleep(5)
    x.save()

    




