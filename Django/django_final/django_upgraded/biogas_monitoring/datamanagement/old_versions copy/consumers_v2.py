import json
from channels.generic.websocket import WebsocketConsumer
# from channels.consumer import SyncConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
import time
import sqlite3 as sql
import paho.mqtt.client as paho
from paho import mqtt
from .models import Parameters, Machine
# from channels.db import database_sync_to_async
# from asgiref.sync import sync_to_async
import time
from django.contrib.auth.decorators import login_required
from usermanagement.models import Warnings,BiogasMachineModerator,BiogasMachineUser
from .models import Machine
from django.db.models.signals import post_save
from django.dispatch import receiver
# from django import
import paho.mqtt.publish as pl

broker = '7e68437c2d4d4cc185d83eb266d03aaa.s1.eu.hivemq.cloud'
port = 8883
topic = "Sensor_Data"
client_id = 'server_iot1'

class ControlStatus(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send("initiate")
    def get_data(self,id):
        try:
            x=Machine.objects.all().get(MachineID=id)
            timex = list(x.parameters_set.all().order_by('time').values())[-1]['time']
            data={"speed":x.parameters_set.all().filter(time=timex,Id_parameter="g01eles").values("value")[0]["value"],"status":"Running"}
        except:
            self.send("id_false")
            return False
        return data
    def receive(self,text_data):
        data=self.get_data(text_data)
        self.send(json.dumps(data))



def check_authority(request):
    if request.user.username == "admin":
        return "ADMIN"
    if BiogasMachineModerator.objects.get(user=request.user).Active == True:
        return "MODERATOR"
    elif BiogasMachineUser.objects.get(user=request.user).Active == True:
        return "USER"
    else:
        return "UNDEFINED"

# @sync_to_async
# def get_warnings(user):
#     return user.biogasmachineuser.Machines.warnings_set.all()


class ModDataConsumer(WebsocketConsumer):
    def get_data(self,user):
        big_dict={}
        for machine in user.biogasmachinemoderator.Machines.all():
            new_dict = {}
            # print(machine)
            # print(len(list(machine.parameters_set.all().order_by('time').values())))
            timex = list(machine.parameters_set.all().order_by('time').values())[-1]['time']
            rows = machine.parameters_set.all().filter(time=timex).values()
            # print(rows[1]["Id_parameter"][3:6])
            if rows[1]["Id_parameter"][3:6] == "ele":
                new_dict["type"]="ele"
            elif rows[1]["Id_parameter"][3:6] == "env":
                new_dict["type"]="env"
            new_dict["time"]=timex
            for i in rows:
                new_dict[i['Id_parameter']] = {'value':i['value'],'time':i['time']}
            big_dict[str(machine.MachineID)]=new_dict
        return big_dict
    def connect(self):
        self.accept()
        data = self.get_data(self.scope["user"])
        # print(data)
        self.send(json.dumps(data))
    def receive(self, text_data):
        if text_data=='initiate' or text_data=='continue':
            data = self.get_data(self.scope["user"])
            self.send(json.dumps(data))


class notinum(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = 'public_room'
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
        
    # @receiver(post_save, sender = Warnings)
    # @debug_fun
    # async def send_notification(self,**kwargs):
    #     number = self.scope["user"].biogasmachineuser.Machines.warnings_set.all().count
    #     y = self.scope["user"].biogasmachineuser.Machines.warnings_set.all().order_by("-id")[0].WarningContent
    #     self.send(json.dumps({"quantity":x.count,"message":y}))
    #     print("just saved")
    async def send_notification(self, event):
        await self.send("text_data=json.dumps({ 'message': event['message'] })")


        # self.send(json.dumps({"quantity":6,"message":"testing 1 2 3 4"}))

# post_save.connect(notinum().send_notification, sender=Warnings)




# @login_required(login_url="/user/login/")
class warnings(WebsocketConsumer):
    def get_data(self,user):
        warnings = user.biogasmachineuser.Machines.warnings_set.all()
        return warnings
    def connect(self):
        self.accept()
        warnings = self.get_data(self.scope["user"])
        # print(warnings)
        i=0
        x={}
        for warning in warnings:
            x[str(i)]= {"type":warning.WarningType,"content":warning.WarningContent}
            i+=1
        # print(x)
        self.send(json.dumps(x))
        
    
def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties=None):    
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
        print(rc)
    client = paho.Client(paho.CallbackAPIVersion.VERSION1,client_id="server_iot1", userdata=None, protocol=paho.MQTTv5)
    client.on_connect = on_connect
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set("binhtay1303", "13032002")
    client.connect(broker, port)
    return client



class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
    def receive(self, text_data):
        if text_data=='initiate' or text_data=='continue':
            data = self.get_data()
            self.send(json.dumps(data))
    def disconnect(self, close_code):
        self.close()
    def get_data(self):
        new_dict = {}
        timex = list(Parameters.objects.order_by('time').values())[-1]['time']
        rows = Parameters.objects.filter(time=timex).values()
        # print(rows[1]["Id_parameter"][3:6])
        if rows[1]["Id_parameter"][3:6] == "ele":
            new_dict["type"]="ele"
        elif rows[1]["Id_parameter"][3:6] == "env":
            new_dict["type"]="env"
        new_dict["time"]=timex
        for i in rows:
            new_dict[i['Id_parameter']] = {'value':i['value'],'time':i['time']}
        return new_dict

class Chatchat(WebsocketConsumer):
    machine = None
    def connect(self):
        self.accept()
        self.machine = self.scope['url_route']['kwargs']['mid']
    def receive(self, text_data):
        if text_data=='initiate' or text_data=='continue':
            data = self.get_data(self.machine)
            self.send(json.dumps(data))
    def disconnect(self, close_code):
        self.close()
    def get_data(self,mname):
        new_dict = {}
        # machine_ins = print(self.scope['url_route']['kwargs']['mid'])
        timex = list(Parameters.objects.filter(MachineIDString = mname).order_by('time').values())[-1]['time']
        rows = Parameters.objects.filter(time=timex).values()
        # print(rows[1]["Id_parameter"][3:6])
        if rows[1]["Id_parameter"][3:6] == "ele":
            new_dict["type"]="ele"
        elif rows[1]["Id_parameter"][3:6] == "env":
            new_dict["type"]="env"
        new_dict["time"]=timex
        for i in rows:
            new_dict[i['Id_parameter']] = {'value':i['value'],'time':i['time']}
        return new_dict








        
