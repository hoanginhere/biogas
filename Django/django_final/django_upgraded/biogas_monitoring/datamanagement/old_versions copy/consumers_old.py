import json
from channels.generic.websocket import WebsocketConsumer
# from channels.consumer import SyncConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
import time
import sqlite3 as sql
from paho.mqtt import client as mqtt_client
from .models import Parameters, Machine
# from channels.db import database_sync_to_async
# from asgiref.sync import sync_to_async
import time
from django.contrib.auth.decorators import login_required
from usermanagement.models import Warnings,BiogasMachineModerator,BiogasMachineUser
from django.db.models.signals import post_save
from django.dispatch import receiver
# from django import


broker='mqtt.eclipseprojects.io'
client_id='sos114'
#username='caseinova'
#password='000000000t'
port = 1883
topic='sensor_data'

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



# def debug_fun(f,sender,**kwargs):
#     print("just saved something")

# post_save.connect(debug_fun, sender=Warnings)

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
        # self.send(json.dumps({"username":self.scope["user"].biogasmachineuser.Machines}))
        # self.send(json.dumps(get_warnings(self.scope["user"])))
        # print(self.get_data(self.scope["user"]))
        warnings = self.get_data(self.scope["user"])
        print(warnings)
        i=0
        x={}
        for warning in warnings:
            x[str(i)]= {"type":warning.WarningType,"content":warning.WarningContent}
            i+=1
        print(x)
        self.send(json.dumps(x))
        # self.send(json.dumps({"messages":str(warnings)}))
        
    



def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client,msg):
    #  msg_count = 1
    #  while True:
    # time.sleep(1)
    # msg = f"messages: {msg_count}"
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        succ="success"
    else:
        succ="failure"
    # msg_count += 1
    # if msg_count > 5:
    #     break
    return succ


class ControllerConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send("success_control")
    def receive(self,text_data):
        mq_client = connect_mqtt()
        stat=publish(mq_client, text_data)
        self.send(stat)


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
        print(rows[1]["Id_parameter"][3:6])
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
        print(rows[1]["Id_parameter"][3:6])
        if rows[1]["Id_parameter"][3:6] == "ele":
            new_dict["type"]="ele"
        elif rows[1]["Id_parameter"][3:6] == "env":
            new_dict["type"]="env"
        new_dict["time"]=timex
        for i in rows:
            new_dict[i['Id_parameter']] = {'value':i['value'],'time':i['time']}
        return new_dict








        
