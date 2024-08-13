from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from datamanagement.models import *
from django.utils.safestring import mark_safe
import json
from usermanagement.models import *
from django.contrib.auth.decorators import login_required


def registration_wall(request):
    if request.user.username == 'admin':
        return True
    try:
        user_logged =  BiogasMachineUser.objects.get(user = request.user)
        mod_logged = BiogasMachineModerator.objects.get(user = request.user)
    except:
        return HttpResponseRedirect('/user/login/')
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

@login_required(login_url="/user/login/")
def landing(request):
    if not registration_wall(request):
        return HttpResponseRedirect('/user/verify/')
    if request.user.is_authenticated:
        return HttpResponseRedirect('/home/')
    else:
        return HttpResponseRedirect('/user/login/')

@login_required(login_url="/user/login/")
def homepage(request):
    if not registration_wall(request):
        return HttpResponseRedirect('/user/verify/')
    usertype = check_authority(request)
    if usertype == "USER":
        try:
            machine_name = request.user.biogasmachineuser.Machines
        except:
            return HttpResponseRedirect('/user/edit_profile/')
        # print(request.user.biogasmachineuser.Machines.MachineID)
        return render(request,'person.html',{"user":request.user,"machine_name":machine_name,"usertype":"USER"})
    else:
        # try:
        if (request.user.username != "admin"):
            machine_name = request.user.biogasmachinemoderator.Machines.all()
        else:
            machine_name="null"
        print(machine_name)
        # except:
            # return HttpResponseRedirect('/user/edit_profile/')
        return render(request,'person.html',{"user":request.user,"usertype":"MODERATOR"})


