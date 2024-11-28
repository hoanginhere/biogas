from django.db import models

# Create your models here.

class Machine(models.Model):
    MachineName = models.CharField(max_length = 255)
    MachineID = models.CharField(max_length = 10, null = True ,unique=True)
    def __str__(self):
        return self.MachineName

class Parameters(models.Model):
    MachineID = models.ForeignKey(Machine, on_delete = models.CASCADE, null=True, blank = True)
    MachineIDString = models.CharField(max_length = 100, blank = True)
    Id_parameter = models.CharField(max_length=100,blank = True)
    value = models.FloatField(null=True)
    time = models.FloatField()
    def __str__(self):
        return self.MachineIDString

class Thresholds(models.Model):
    MachineId = models.CharField(max_length=100)
    ParamID = models.CharField(max_length=100)
    Value = models.FloatField()

    