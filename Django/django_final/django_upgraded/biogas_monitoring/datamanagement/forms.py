from django import forms
from django.forms import ModelForm
from .models import *


class AddMachine(ModelForm):
    class Meta:
        model = Machine
        fields = ["MachineName","MachineID"]

class SelectMachine(forms.Form):
    MachineName = forms.CharField(max_length = 20)


class ControlForm(forms.Form):
    choices=[("SPEED","speed control"),("POW","turn on or off")]
    ControlSignal = forms.ChoiceField(choices=choices)
    MachineID = forms.CharField(max_length=20)
    Param = forms.FloatField()

class VibrationForm(forms.Form):
    TimeFrame = forms.FloatField()
    MachineID = forms.CharField(max_length=20)

class DateInterval(forms.Form):
    MachineID = forms.CharField(max_length=20)
    TimeStart = forms.DateTimeField(widget=forms.SelectDateWidget())
    TimeEnd = forms.DateTimeField(widget=forms.SelectDateWidget())

class ThresholdForm(ModelForm):
    class Meta:
        model = Thresholds
        fields = ["MachineId","ParamID","Value"]


    
    