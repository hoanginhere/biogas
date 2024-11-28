from django.db import models
from datamanagement.models import *
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator

# Create your models here.

class BiogasMachineModerator(models.Model):
    MALE = "MALE"
    FEMALE = "FEMALE"
    UNDEFINED = "UNDEFINED"
    GENDER_CHOICES = [(MALE,"Male"),(FEMALE,"Female"),(UNDEFINED,"Undefined")]
    Machines = models.ManyToManyField(Machine)
    user = models.OneToOneField(User, on_delete = models.CASCADE,null=True)
    PhoneNumber = models.IntegerField(null= True)
    Sex = models.CharField(max_length = 9, choices = GENDER_CHOICES, default = UNDEFINED)
    Address = models.CharField(max_length = 255,blank = True, null=True)
    Registered = models.BooleanField(default = False)
    Active = models.BooleanField(default = False)

class BiogasMachineUser(models.Model):
    MALE = "MALE"
    FEMALE = "FEMALE"
    UNDEFINED = "UNDEFINED"
    GENDER_CHOICES = [(MALE,"Male"),(FEMALE,"Female"),(UNDEFINED,"Undefined")]
    Machines = models.OneToOneField(Machine, on_delete = models.CASCADE,null=True, blank = True)
    user = models.OneToOneField(User, on_delete = models.CASCADE,null=True)
    PhoneNumber = models.IntegerField(null= True)
    Sex = models.CharField(max_length = 9, choices = GENDER_CHOICES, default = UNDEFINED)
    Address = models.CharField(max_length = 255,blank = True, null=True)
    Registered = models.BooleanField(default = False)
    Active = models.BooleanField(default = False)

    
class RegistrationCode(models.Model):
    MOD = "MODERATOR"
    USER = "REGULAR"
    USER_TYPE = [(MOD,"Moderator"),(USER, "Regular user")]
    Code = models.CharField(max_length = 30)
    Value = models.BooleanField()
    UserType = models.CharField(max_length=9,choices = USER_TYPE, null=True, blank=True)


class Warnings(models.Model):
    WarningChoices = [("ELE","Electrical"),("ENV","Environment"),("OPE", "Operational"),("MISC", "Miscellaneous")]
    StatusType = [("T","Resolved"),("F","Unresolved")]
    WarningContent = models.CharField(max_length = 255)
    WarningType = models.CharField(max_length=11,choices=WarningChoices, default = "MISC")
    WarningResolution = models.CharField(max_length=2,choices=StatusType,default="F")
    Machines = models.ForeignKey(Machine, on_delete= models.CASCADE,null=True)
    def __str__(self):
            return self.WarningContent


