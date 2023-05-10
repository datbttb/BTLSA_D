from  rest_framework import serializers
from  image_model.models import Image

class ImageSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(required=False)
    class Meta:
        model = Image
        fields = '__all__'