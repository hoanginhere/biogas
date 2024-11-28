from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import *
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .models import *
from django.core.exceptions import ObjectDoesNotExist
import random
import string
from django.contrib.auth.decorators import login_required
from datamanagement.models import Machine
from django.views.generic import DetailView


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
    if BiogasMachineModerator.objects.get(user=request.user).Active == True:
        return "MODERATOR"
    elif BiogasMachineUser.objects.get(user=request.user).Active == True:
        return "USER"
    else:
        return "UNDEFINED"

# hàm đăng nhập cho user
def login_current_user(request):
    if not request.user.is_authenticated and request.method != 'POST':
        form = LoginForm()
        return render(request,'login.html',{"form":form})
    elif request.user.is_authenticated and request.method != 'POST':
        return HttpResponseRedirect('/home/')
    else:
        user_info = LoginForm(request.POST)
        if user_info.is_valid():
            username = user_info.cleaned_data['username']
            password = user_info.cleaned_data['password']
            current_user = authenticate(request,username=username,password=password)
            if current_user is not None:
                login(request,current_user)
                return HttpResponseRedirect('/home/')
            else:
                error_message = "Your username or password is wrong. Please try again"
                form = LoginForm()
                return render(request, 'login.html',{"form":form,"error_message":error_message})

# hàm đăng xuất cho user
@login_required(login_url='/user/login/')
def logout_current_user(request):
    if request.user.is_authenticated:
        logout(request)
        return HttpResponseRedirect('/user/login/')
    else:
        return HttpResponseRedirect('/user/login/')
def register_new_user(request):
    if request.method != 'POST':
        register_form = UserCreationForm()
        return render(request,'register.html',{'form':register_form})
    else:
        new_user_object = UserCreationForm(request.POST)
        if new_user_object.is_valid():
            new_user_object.save()
            new_user_object.biogasmachineuser = BiogasMachineUser(user = User.objects.get(username=new_user_object.cleaned_data["username"]))
            new_user_object.biogasmachineuser.save()
            new_user_object.biogasmachinemoderator = BiogasMachineModerator(user=User.objects.get(username=new_user_object.cleaned_data["username"]))
            new_user_object.biogasmachinemoderator.save()
            return HttpResponseRedirect('/user/login/')
        else:
            return render(request,'register.html',{"form":"Error Signing up, please try again later"})

@login_required(login_url='/user/login/')
def verify_registration(request):
    if request.method!='POST':
        reg_form = RegistrationCodeForm()
        return render(request,'registration_code_enter.html',{'form':reg_form})
    elif request.method == 'POST':
        reg_data = RegistrationCodeForm(request.POST)
        if reg_data.is_valid():
            # print(1)
            code_data = reg_data.cleaned_data['Code']
            try:
                check_object =  RegistrationCode.objects.get(Code = code_data)
                # print(check_object.UserType)
            except:
                # print(2)
                return render(request,'registration_code_enter.html',{'message':"Wrong registration number",'form':reg_data})
            if check_object!=None:
                # print(3)
                if check_object.Value == True:
                    # print(4)
                    if check_object.UserType == "MODERATOR":
                        # print(9)
                        mod_object_mod = BiogasMachineModerator.objects.get(user = request.user)
                        mod_object_mod.Registered = True
                        mod_object_mod.Active = True
                        mod_object_mod.save()
                        check_object.Value = False
                        check_object.save()
                        return HttpResponseRedirect('/home/')
                    elif check_object.UserType == "REGULAR":
                        # print(5)
                        user_object_user = BiogasMachineUser.objects.get(user = request.user)
                        user_object_user.Registered = True
                        user_object_user.Active = True
                        user_object_user.save()
                        check_object.Value = False
                        check_object.save()
                        return HttpResponseRedirect('/home/')
                else:
                    # print(6)
                    form=RegistrationCodeForm()
                    return render(request,'registration_code_enter.html',{'form':form,'message':"This registration has already been used"})
            else:
                # print(7)
                return render(request,'registration_code_enter.html',{'message':"Wrong registration number"})
        else:
            # print(8)
            return HttpResponseRedirect('/home/')


# hàm chỉnh sửa profile của người dùng
@login_required(login_url='/user/login/')
def edit_user_profile(request):
    # print(check_authority(request))
    if check_authority(request) == "USER":
        if request.method != 'POST':
            form = UserProfileEdit()
            return render(request,'user_profile_edit.html',{"form":form})
        else:
            current_user = User.objects.get(username = request.user.username)
            form = UserProfileEdit(request.POST, instance = current_user.biogasmachineuser)
            form.save()
            message = "Updated user profile successfully"
            return render(request,'user_profile_edit.html',{"message":message})
    elif check_authority(request) == "MODERATOR":
        if request.method != 'POST':
            form = ModeratorProfileEdit()
            return render(request,'user_profile_edit.html',{"form":form})
        else:
            current_user = User.objects.get(username = request.user.username)
            machines=request.POST.getlist('Machines')
            # print(machines)
            form = UserProfileEdit(request.POST, instance = current_user.biogasmachinemoderator)
            form.is_valid()
            instance_mod = form.save(commit=False)
            for i in machines:
                instance_mod.Machines.add(Machine.objects.get(id=int(i)))
            instance_mod.save()
            message = "Updated user profile successfully"
            
            return render(request,'user_profile_edit.html',{"message":message})



# hàm tạo registration code
@login_required(login_url='/user/login/')
def generate_registration_code(request):
    if request.user.username == 'admin':
        if request.method != 'POST':
            form = number_form()
            return render(request,'registration_code_generator.html',{"form":form})
        else:
            number = int(request.POST['number'])
            code_list=''
            for i in range(int(number/2)):
                letters = string.ascii_letters
                x =  ''.join(random.choice(letters) for i in range(26))
                code = RegistrationCode(Code = x, Value = True, UserType = "REGULAR")
                code_list+= x+'\n'
                code.save()
            for i in range(int(number/2)):
                letters = string.ascii_letters
                x =  ''.join(random.choice(letters) for i in range(26))
                code = RegistrationCode(Code = x, Value = True, UserType = "MODERATOR")
                code_list+= x+'\n'
                code.save()
            return render(request,'registration_code_generator.html',{"message":"Successfully created "+str(number)+" codes:"})
    else:
        return render(request,'registration_code_generator.html',{"form":"You do not have this permission"})


def ProfileListView(request):
    x=check_authority(request)
    if x=="MODERATOR":
        user_object=request.user.biogasmachinemoderator
        return render(request,"profile_view.html",{"sex":user_object.Sex,"machines":user_object.Machines,"phonenumber":user_object.PhoneNumber,"address":user_object.Address,"registered":user_object.Registered,"name":user_object.user.username})
    elif x=="USER":
        user_object=request.user.biogasmachineuser
        return render(request,"profile_view.html",{"sex":user_object.Sex,"machines":user_object.Machines,"phonenumber":user_object.PhoneNumber,"address":user_object.Address,"registered":user_object.Registered,"name":user_object.user.username})


