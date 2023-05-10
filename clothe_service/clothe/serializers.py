from  rest_framework import serializers
from  clothe.models import Clothe

class ClotheSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clothe
        fields = '__all__'