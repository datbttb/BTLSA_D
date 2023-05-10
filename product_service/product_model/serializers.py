from  rest_framework import serializers
from  product_model.models import product_details

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = product_details
        fields = '__all__'