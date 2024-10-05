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

from django import forms

# Định nghĩa danh sách thiết bị
DEVICES = [
    ('exhaust_fan', 'Exhaust Fan'),
    ('wastewater_treatment', 'Wastewater Treatment'),
    ('cooling_pump', 'Cooling Pump'),
    ('lighting', 'Lighting'),
    ('aeration', 'Aeration'),
    ('compensation_pump', 'Compensation Pump'),
]

from django import forms

from django import forms

class ScheduleForm(forms.Form):
    # Form cho exhaust fan
    exhaust_fan_schedule_option = forms.ChoiceField(
        choices=[("yes", "Có lập lịch"), ("no", "Không lập lịch")],
        label="Lập lịch cho Quạt hút?",
        required=False
    )
    exhaust_fan_run_time = forms.ChoiceField(
        choices=[
            ("day", "Chạy ngày"),
            ("night", "Chạy đêm"),
            (None, "Không chạy"),
            ("full_day", "Chạy cả ngày"),
        ],
        label="Chế độ chạy không lập lịch",
        required=False,
    )
    exhaust_fan_schedule = forms.ChoiceField(
        choices=[
            ("day", "Lập lịch sáng"),
            ("night", "Lập lịch đêm"),
            ("full_day", "Lập lịch cả ngày"),
        ],
        label="Chọn lịch trình cho Quạt hút",
        required=False,
    )
    exhaust_fan_exact_hours = forms.IntegerField(
        label="Chạy đúng giờ cho Quạt hút",
        required=False
    )
    exhaust_fan_power_mode = forms.ChoiceField(
        choices=[
            ("half", "Nửa công suất"),
            ("full", "Toàn bộ công suất"),
        ],
        label="Chọn công suất cho Quạt hút",
        required=False,
    )

    # Form cho wastewater treatment
    wastewater_treatment_schedule_option = forms.ChoiceField(
        choices=[("yes", "Có lập lịch"), ("no", "Không lập lịch")],
        label="Lập lịch cho Xử lý nước thải?",
        required=False
    )
    wastewater_treatment_run_time = forms.ChoiceField(
        choices=[
            ("day", "Chạy ngày"),
            ("night", "Chạy đêm"),
            (None, "Không chạy"),
            ("full_day", "Chạy cả ngày"),
        ],
        label="Chế độ chạy không lập lịch",
        required=False,
    )
    wastewater_treatment_schedule = forms.ChoiceField(
        choices=[
            ("day", "Lập lịch sáng"),
            ("night", "Lập lịch đêm"),
            ("full_day", "Lập lịch cả ngày"),
        ],
        label="Chọn lịch trình cho Xử lý nước thải",
        required=False,
    )
    wastewater_treatment_exact_hours = forms.IntegerField(
        label="Chạy đúng giờ cho Xử lý nước thải",
        required=False
    )
    wastewater_treatment_power_mode = forms.ChoiceField(
        choices=[
            ("half", "Nửa công suất"),
            ("full", "Toàn bộ công suất"),
        ],
        label="Chọn công suất cho Xử lý nước thải",
        required=False,
    )

    # Form cho cooling pump
    cooling_pump_schedule_option = forms.ChoiceField(
        choices=[("yes", "Có lập lịch"), ("no", "Không lập lịch")],
        label="Lập lịch cho Máy bơm làm mát?",
        required=False
    )
    cooling_pump_run_time = forms.ChoiceField(
        choices=[
            ("day", "Chạy ngày"),
            ("night", "Chạy đêm"),
            (None, "Không chạy"),
            ("full_day", "Chạy cả ngày"),
        ],
        label="Chế độ chạy không lập lịch",
        required=False,
    )
    cooling_pump_schedule = forms.ChoiceField(
        choices=[
            ("day", "Lập lịch sáng"),
            ("night", "Lập lịch đêm"),
            ("full_day", "Lập lịch cả ngày"),
        ],
        label="Chọn lịch trình cho Máy bơm làm mát",
        required=False,
    )
    cooling_pump_exact_hours = forms.IntegerField(
        label="Chạy đúng giờ cho Máy bơm làm mát",
        required=False
    )
    cooling_pump_power_mode = forms.ChoiceField(
        choices=[
            ("half", "Nửa công suất"),
            ("full", "Toàn bộ công suất"),
        ],
        label="Chọn công suất cho Máy bơm làm mát",
        required=False,
    )

    # Form cho lighting
    lighting_schedule_option = forms.ChoiceField(
        choices=[("yes", "Có lập lịch"), ("no", "Không lập lịch")],
        label="Lập lịch cho Đèn chiếu sáng?",
        required=False
    )
    lighting_run_time = forms.ChoiceField(
        choices=[
            ("day", "Chạy ngày"),
            ("night", "Chạy đêm"),
            (None, "Không chạy"),
            ("full_day", "Chạy cả ngày"),
        ],
        label="Chế độ chạy không lập lịch",
        required=False,
    )
    lighting_schedule = forms.ChoiceField(
        choices=[
            ("day", "Lập lịch sáng"),
            ("night", "Lập lịch đêm"),
            ("full_day", "Lập lịch cả ngày"),
        ],
        label="Chọn lịch trình cho Đèn chiếu sáng",
        required=False,
    )
    lighting_exact_hours = forms.IntegerField(
        label="Chạy đúng giờ cho Đèn chiếu sáng",
        required=False
    )
    lighting_power_mode = forms.ChoiceField(
        choices=[
            ("half", "Nửa công suất"),
            ("full", "Toàn bộ công suất"),
        ],
        label="Chọn công suất cho Đèn chiếu sáng",
        required=False,
    )

    # Form cho aeration
    aeration_schedule_option = forms.ChoiceField(
        choices=[("yes", "Có lập lịch"), ("no", "Không lập lịch")],
        label="Lập lịch cho Máy sục khí?",
        required=False
    )
    aeration_run_time = forms.ChoiceField(
        choices=[
            ("day", "Chạy ngày"),
            ("night", "Chạy đêm"),
            (None, "Không chạy"),
            ("full_day", "Chạy cả ngày"),
        ],
        label="Chế độ chạy không lập lịch",
        required=False,
    )
    aeration_schedule = forms.ChoiceField(
        choices=[
            ("day", "Lập lịch sáng"),
            ("night", "Lập lịch đêm"),
            ("full_day", "Lập lịch cả ngày"),
        ],
        label="Chọn lịch trình cho Máy sục khí",
        required=False,
    )
    aeration_exact_hours = forms.IntegerField(
        label="Chạy đúng giờ cho Máy sục khí",
        required=False
    )
    aeration_power_mode = forms.ChoiceField(
        choices=[
            ("half", "Nửa công suất"),
            ("full", "Toàn bộ công suất"),
        ],
        label="Chọn công suất cho Máy sục khí",
        required=False,
    )

    # Form cho compensation pump
    compensation_pump_schedule_option = forms.ChoiceField(
        choices=[("yes", "Có lập lịch"), ("no", "Không lập lịch")],
        label="Lập lịch cho Máy bơm bù?",
        required=False
    )
    compensation_pump_run_time = forms.ChoiceField(
        choices=[
            ("day", "Chạy ngày"),
            ("night", "Chạy đêm"),
            (None, "Không chạy"),
            ("full_day", "Chạy cả ngày"),
        ],
        label="Chế độ chạy không lập lịch",
        required=False,
    )
    compensation_pump_schedule = forms.ChoiceField(
        choices=[
            ("day", "Lập lịch sáng"),
            ("night", "Lập lịch đêm"),
            ("full_day", "Lập lịch cả ngày"),
        ],
        label="Chọn lịch trình cho Máy bơm bù",
        required=False,
    )
    compensation_pump_exact_hours = forms.IntegerField(
        label="Chạy đúng giờ cho Máy bơm bù",
        required=False
    )
    compensation_pump_power_mode = forms.ChoiceField(
        choices=[
            ("half", "Nửa công suất"),
            ("full", "Toàn bộ công suất"),
        ],
        label="Chọn công suất cho Máy bơm bù",
        required=False,
    )
