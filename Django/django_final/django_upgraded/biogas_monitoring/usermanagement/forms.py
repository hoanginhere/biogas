from django import forms
from django.forms import ModelForm
from .models import BiogasMachineUser, RegistrationCode, BiogasMachineModerator
from datamanagement.models import Machine


class LoginForm(forms.Form):
    username = forms.CharField(max_length=30)
    password = forms.CharField(widget=forms.PasswordInput,max_length=30,)

class UserProfileEdit(ModelForm):
    class Meta:
        model = BiogasMachineUser
        fields = ["Sex","Address","Machines","PhoneNumber"]
class ModeratorProfileEdit(ModelForm):
    Machines = forms.ModelMultipleChoiceField(
                        queryset=Machine.objects.all(),
                        label="Machines",
                        widget=forms.CheckboxSelectMultiple)

    class Meta:
        model = BiogasMachineModerator
        fields = ["Sex","Address","Machines","PhoneNumber"]


class RegistrationCodeForm(ModelForm):
    class Meta:
        model = RegistrationCode
        fields = ["Code"]

class number_form(forms.Form):
    number = forms.IntegerField()





