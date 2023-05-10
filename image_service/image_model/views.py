from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response

from image_model.models import Image
from image_model.serializers import ImageSerializer


class ImageViewSet(viewsets.ViewSet):
    def list(self, request):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer= ImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def retrieve(self, request, pk=None):
        product = Image.objects.get(id=pk)
        serializer = ImageSerializer(product)
        return Response(serializer.data)


    def update(self, request, pk=None):
        product = Image.objects.get(id=pk)
        serializer = ImageSerializer(instance=product, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(status=status.HTTP_202_ACCEPTED)

    def destroy(self, request, pk=None):
        product = Image.objects.get(id=pk)
        product.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
