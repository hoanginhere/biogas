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

broker = '7e68437c2d4d4cc185d83eb266d03aaa.s1.eu.hivemq.cloud'
port = 8883
topic = "Sensor_Data"
client_id = 'server_iot'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):    
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = paho.Client(paho.CallbackAPIVersion.VERSION1,client_id="server_iot", userdata=None, protocol=paho.MQTTv5)
    client.on_connect = on_connect
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set("binhtay1303", "13032002")
    
    client.connect(broker, port)
    return client


def on_message(client, userdata, msg):
    global data
    global time
    data = str(msg.payload.decode())
    data_dict=json.loads(data)
    # time=data_dict["time"]
    gen_name = data_dict['rpi']
    # print(gen_name)
    generator = Machine.objects.get(MachineID=gen_name)
    for key,val in data_dict['data'].items():
        new_col = Parameters(MachineID=generator,MachineIDString=gen_name,Id_parameter=key,value=val,time=data_dict["time"])
        # print('saving'+str(new_col))
        new_col.save()

client = connect_mqtt()
client.subscribe(topic)
client.on_message=on_message

client.loop_forever()

