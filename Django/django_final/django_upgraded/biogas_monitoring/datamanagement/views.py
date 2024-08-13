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

broker = '27.71.16.120'
port = 1883
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
    print(request.method)
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
        # form = SelectMachine()
        # return render(request,'industrial.html',{"usertype":author,"form":form,"machine":"common"})
        if author=="MODERATOR":
            return render(request,'industrial.html',{"usertype":author,"alert":"Please choose a biogas generator to monitor"})
        return render(request,'industrial.html',{"usertype":author,"ws_machine":"common"})
    elif request.method == "POST":
        form = request.POST
        # print(form)
        # form = SelectMachine(request.POST)
        # if form.is_valid():
        machine_name = form['Machine']
        # print(machine_name)
        try:
            machine_ins = Machine.objects.get(MachineID = machine_name)
        except:
            return render(request,'industrial.html',{"usertype":author,"machine":"Machine "+machine_name+" does not exist"})
        return HttpResponseRedirect('/data/industrial/'+machine_name+'/')
        # return render(request,'industrial.html',{"usertype":author,"machine":machine_name})
@login_required(login_url="/user/login/")

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
            pl.single(topic_control,payload=payload,hostname=broker,port=port,client_id=client_id,keepalive=60)
            if (control_form.cleaned_data["ControlSignal"]=="POW"):
                if (control_form.cleaned_data["Param"]==1):
                    status_sp_string = "Running"
                if (control_form.cleaned_data["Param"]==0):
                    status_sp_string = "Stopped"
            else:
                status_sp_string="None"
            return HttpResponseRedirect('/data/controller/'+control_form.cleaned_data['MachineID']+"/"+status_sp_string)
        else:
            print(control_form)
@login_required(login_url="/user/login/")
def controller_view_monitor(request,machine,status_sp):
    if request.method != "POST":
        control_form=ControlForm()
        return render(request,'controller_monitor.html',{"id":machine,"form_send":control_form,"status_sp":status_sp})
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
            pl.single(topic_control,payload=payload,hostname=broker,port=port,client_id=client_id,keepalive=60)
            if (control_form.cleaned_data["ControlSignal"]=="POW"):
                if (control_form.cleaned_data["Param"]==1):
                    status_sp_string = "Running"
                if (control_form.cleaned_data["Param"]==0):
                    status_sp_string = "Stopped"
            else:
                status_sp_string="None"
            return HttpResponseRedirect('/data/controller/'+control_form.cleaned_data['MachineID']+'/'+status_sp_string)
        else:
            print("invalid")
@login_required(login_url="/user/login/")
def industrial_gui_1(request,mid):
    if not registration_wall(request):
        return HttpResponseRedirect('/user/verify/')
    author=check_authority(request)
    if request.method != "POST":
        return render(request,'industrial.html',{"usertype":author,"ws_machine":mid,"machine_name":Machine.objects.get(MachineID=mid)})
    elif request.method == "POST":
        form = request.POST
        machine_name = form['Machine']
        # print(machine_name)
        try:
            machine_ins = Machine.objects.get(MachineID = machine_name)
        except:
            return render(request,'industrial.html',{"usertype":author,"Error_code":"Machine "+machine_name+" does not exist"})
        return HttpResponseRedirect('/data/industrial/'+machine_name+'/')

@login_required(login_url="/user/login/")
def loadgraph(request):
    if request.method=="POST":
        return render(request,"loadgraph.html",{"status":"success","form_content":request.POST})
    return render(request,"loadgraph.html")

@login_required(login_url="/user/login/")
def interval(request):
    author = check_authority(request)
    if request.method=="POST":
        form= DateInterval(request.POST)
        if form.is_valid():
            id=form.cleaned_data["MachineID"]
            if check_authority(request) == "MODERATOR":
                if not request.user.biogasmachinemoderator.Machines.all().filter(MachineID=form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            if check_authority(request) == "USER":
                if not Machine.objects.filter(biogasmachineuser=request.user.biogasmachineuser,MachineID=form.cleaned_data["MachineID"]):
                    return render(request,"401.html")
            print(form.cleaned_data["TimeStart"])
            return render(request,"interval.html",{"status":"success", "id":id,"ts":form.cleaned_data["TimeStart"].timestamp(),"te":form.cleaned_data["TimeEnd"].timestamp(),"form_query":DateInterval()})
    form = DateInterval()
    return render(request,"interval.html",{"form_query":form})

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
            pl.single(topic_vibration,payload=payload,hostname=broker,port=port,client_id=client_id,keepalive=60)
            # return render(request,"vibration.html",{"status":(str(vibration_form.cleaned_data["MachineID"])+'\n'+str(vibration_form.cleaned_data["TimeFrame"]))})
            return HttpResponseRedirect("/data/vibration/result/"+vibration_form.cleaned_data["MachineID"])

@login_required(login_url="/user/login/")
def vibration_result(request,machine):
    return render(request,"vibration_result.html",{"machine":machine})


@login_required(login_url="/user/login/")
def threshold(request):
    if request.method=="POST":
        form = ThresholdForm(request.POST)
        form.save()
        return HttpResponseRedirect("/data/industrial/")

# Create your views here.
