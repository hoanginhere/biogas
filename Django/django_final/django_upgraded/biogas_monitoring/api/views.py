from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import warningSerializer
from usermanagement.models import Warnings

@api_view(['GET'])
def getData(request):
    person = {'name':'Denis','age':28}
    warningss = Warnings.objects.all()
    serializer = warningSerializer(warningss,many=True)
    return Response(serializer.data)

