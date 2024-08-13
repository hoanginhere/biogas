from rest_framework import serializers
from usermanagement.models import Warnings


class warningSerializer(serializers.ModelSerializer):
    class Meta:
        model = Warnings
        fields = '__all__'