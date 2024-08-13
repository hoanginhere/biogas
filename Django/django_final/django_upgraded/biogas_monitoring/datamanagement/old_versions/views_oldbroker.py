from django.shortcuts import render
from .models import *
from .forms import *
from usermanagement.models import *
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
import paho.mqtt.publish as pl
from paho import mqtt
import paho.mqtt.client as paho
import json

broker = '7e68437c2d4d4cc185d83eb266d03aaa.s1.eu.hivemq.cloud'
port = 8883
topic = "Sensor_Data"
topic_control = "Control_Data"
topic_vibration = "Vibration_Data"
client_id = 'server_iot1'


def registration_wall(request):
    if request.user.username == 'admin':
        return True
    user_logged =  BiogasMachineUser.objects.get(user = request.user)
    mod_logged = BiogasMachineModerator.objects.get(user = request.user)
    if user_logged!=None or mod_logged!=None:
        if user_logged.Registered == False and mod_logged.Registered == False:
            return False
        else:
            return True
    else:
        return True


def check_authority(request):
    if request.user.username == "admin":
        return "ADMIN"
    if BiogasMachineModerator.objects.get(user=request.user).Active == True:
        return "MODERATOR"
    elif BiogasMachineUser.objects.get(user=request.user).Active == True:
        return "USER"
    else:
        return "UNDEFINED"

#ham warning
@login_required(login_url="/user/login/")
def warning_view(request):
    return render(request,'warnings.html')


# hàm thêm máy phát biogas vào cơ sở dữ liệu
@login_required(login_url="/user/login/")
def add_machine(request):
    if request.user.username == 'admin':
        if request.method == 'POST':
            current_machine = AddMachine(request.POST)
            if current_machine.is_valid():
                current_machine.save()
                return render(request,'add_machine.html',{"message":"Successfully added machine"})
            else:
                form=AddMachine()
                return render(request,'add_machine.html',{"message":"Machine id already existed, type another one", "form":form})
        else:
            form = AddMachine()
            return render(request,'add_machine.html',{'form':form})
    elif request.user.username != 'admin':
        return render(request,'401.html')

@login_required(login_url="/user/login/")
def industrial_gui(request):
    if not registration_wall(request):
        return HttpResponseRedirect('/user/verify/')
    author=check_authority(request)
    if request.method != "POST":
        form = SelectMachine()
        return render(request,'industrial.html',{"usertype":author,"form":form,"machine":"common"})
    elif request.method == "POST":
        form = SelectMachine(request.POST)
        if form.is_valid():
            machine_name = form.cleaned_data['MachineName']
        try:
            machine_ins = Machine.objects.get(MachineID = machine_name)
        except:
            return render(request,'industrial.html',{"usertype":author,"machine":"Machine "+machine_name+" does not exist"})
        return HttpResponseRedirect('/data/industrial/'+machine_name+'/')
        # return render(request,'industrial.html',{"usertype":author,"machine":machine_name})

@login_required(login_url="/user/login/")
def controller_view(request):
    if request.method != "POST":
        control_form = ControlForm(initial={"ControlSignal":"POW"})
        return render(request, 'controller.html', {"form_send":control_form})
    if request.method == "POST":
        control_form = ControlForm(request.POST)
        if control_form.is_valid():
            if check_authority(request) == "MODERATOR":
                if not request.user.biogasmachinemoderator.Machines.all().filter(MachineID=control_form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            if check_authority(request) == "USER":
                if not Machine.objects.filter(biogasmachineuser=request.user.biogasmachineuser,MachineID=control_form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            payload = json.dumps({"id":control_form.cleaned_data["MachineID"],"command":control_form.cleaned_data["ControlSignal"],"param":control_form.cleaned_data["Param"]})
            pl.single(topic_control,payload=payload,hostname=broker,port=port,client_id=client_id,keepalive=60,auth={"username":"binhtay1303","password":"13032002"},tls={'tls_version':mqtt.client.ssl.PROTOCOL_TLS},protocol=paho.MQTTv5)
            return HttpResponseRedirect('/data/controller/'+control_form.cleaned_data['MachineID'])
        else:
            print(control_form)
@login_required(login_url="/user/login/")
def controller_view_monitor(request,machine):
    if request.method != "POST":
        control_form=ControlForm()
        return render(request,'controller_monitor.html',{"id":machine,"form_send":control_form})
    if request.method == "POST":
        control_form = ControlForm(request.POST)
        if control_form.is_valid():
            if check_authority(request) == "MODERATOR":
                if not request.user.biogasmachinemoderator.Machines.all().filter(MachineID=control_form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            if check_authority(request) == "USER":
                if not Machine.objects.filter(biogasmachineuser=request.user.biogasmachineuser,MachineID=control_form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            payload = json.dumps({"id":control_form.cleaned_data["MachineID"],"command":control_form.cleaned_data["ControlSignal"],"param":control_form.cleaned_data["Param"]})
            pl.single(topic_control,payload=payload,hostname=broker,port=port,client_id=client_id,keepalive=60,auth={"username":"binhtay1303","password":"13032002"},tls={'tls_version':mqtt.client.ssl.PROTOCOL_TLS},protocol=paho.MQTTv5)
            return HttpResponseRedirect('/data/controller/'+control_form.cleaned_data['MachineID'])




@login_required(login_url="/user/login/")
def industrial_gui_1(request,mid):
    if not registration_wall(request):
        return HttpResponseRedirect('/user/verify/')
    return render(request,'industrial.html',{"machine":mid})

@login_required(login_url="/user/login/")
def loadgraph(request):
    if request.method=="POST":
        return render(request,"loadgraph.html",{"status":"success","form_content":request.POST})
    return render(request,"loadgraph.html")

@login_required(login_url="/user/login/")
def interval(request):
    if request.method=="POST":
        return render(request,"interval.html",{"status":"success", "form_content":request.POST})
    return render(request,"interval.html")

@login_required(login_url="/user/login/")
def vibration_view(request):
    if request.method != "POST":
        vibration_form = VibrationForm(initial={"TimeFrame":10.0})
        return render(request, 'vibration.html', {"form_send":vibration_form})
    if request.method == "POST":
        vibration_form = VibrationForm(request.POST)
        if vibration_form.is_valid():
            if check_authority(request) == "MODERATOR":
                if not request.user.biogasmachinemoderator.Machines.all().filter(MachineID=vibration_form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            if check_authority(request) == "USER":
                if not Machine.objects.filter(biogasmachineuser=request.user.biogasmachineuser,MachineID=vibration_form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            payload = json.dumps({"id":vibration_form.cleaned_data["MachineID"],"duration":vibration_form.cleaned_data["TimeFrame"]})
            pl.single(topic_vibration,payload=payload,hostname=broker,port=port,client_id=client_id,keepalive=60,auth={"username":"binhtay1303","password":"13032002"},tls={'tls_version':mqtt.client.ssl.PROTOCOL_TLS},protocol=paho.MQTTv5)
            # return HttpResponseRedirect('/data/vibration/'+vibration_form.cleaned_data['MachineID'])
            return render(request,"vibration.html",{"status":(str(vibration_form.cleaned_data["MachineID"])+'\n'+str(vibration_form.cleaned_data["TimeFrame"]))})

@login_required(login_url="/user/login/")
def vibration_result(request,machine):
    return render(request,"vibration_result.html",{"machine":machine})

# Create your views here.
