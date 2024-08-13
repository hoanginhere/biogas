from django import forms

class InputParameter(forms.Form):
    ParameterID = forms.CharField(max_length=20)