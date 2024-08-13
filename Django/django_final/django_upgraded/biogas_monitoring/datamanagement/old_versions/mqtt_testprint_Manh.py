import paho.mqtt.client as paho
from paho import mqtt
# import os
# import django
# import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
# os.environ['DJANGO_SETTINGS_MODULE'] = 'biogas_monitoring.settings'
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "biogas_monitoring.settings")
# django.setup()
# from datamanagement.models import *
import json


# broker = '7e68437c2d4d4cc185d83eb266d03aaa.s1.eu.hivemq.cloud'
# port = 8883
broker = '27.71.16.120'
port = 1883
topic = "Control_Data"
client_id = 'server_iot'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):    
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # client = paho.Client(paho.CallbackAPIVersion.VERSION1,client_id="server_iot", userdata=None, protocol=paho.MQTTv5)
    client = paho.Client(paho.CallbackAPIVersion.VERSION1,client_id=client_id)
    
    client.on_connect = on_connect
    # client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    # client.username_pw_set("binhtay1303", "13032002")
    client.connect(broker, port)
    return client

def on_message(client, userdata, msg):
    global data
    data = json.loads(str(msg.payload.decode()))
    print(data)

def subscribe(client):
    def on_message(client, userdata, msg):
        global data
        data = json.loads(str(msg.payload.decode()))
        print(data)
    client.subscribe(topic)
    client.on_message=on_message

def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

if __name__=='__main__':
    run()

