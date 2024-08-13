import paho.mqtt.client as paho
from paho import mqtt
import os
import django
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ['DJANGO_SETTINGS_MODULE'] = 'biogas_monitoring.settings'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "biogas_monitoring.settings")
django.setup()
from datamanagement.models import *
import json

broker = '27.71.16.120'
port = 1883
topic = "Sensor_Data"
client_id = 'server_iot'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):    
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = paho.Client(paho.CallbackAPIVersion.VERSION1,client_id="server_iot_biogas")
    client.on_connect = on_connect
    
    client.connect(broker, port)
    return client


def on_message(client, userdata, msg):
    global data
    global time
    count=0
    data = str(msg.payload.decode())
    data_dict=json.loads(data)
    # time=data_dict["time"]
    gen_name = data_dict['rpi']
    print("received "+data)
    try:
        generator = Machine.objects.get(MachineID=gen_name)
    except:
        print("Machine does not exist")
        return 0
    for key,val in data_dict['data'].items():
        new_col = Parameters(MachineID=generator,MachineIDString=gen_name,Id_parameter=key,value=val,time=data_dict["time"])
        count+=1
        new_col.save()
    # print(f'saved {count} values')

client = connect_mqtt()
client.subscribe(topic)
client.on_message=on_message

client.loop_forever()

