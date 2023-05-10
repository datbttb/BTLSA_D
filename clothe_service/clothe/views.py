from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response

from clothe.models import Clothe
from clothe.serializers import ClotheSerializer


class ClotheViewSet(viewsets.ViewSet):
    def list(self, request):
        clothes = Clothe.objects.all()
        serializer = ClotheSerializer(clothes, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer= ClotheSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def retrieve(self, request, pk=None):
        clothe = Clothe.objects.get(id=pk)
        serializer = ClotheSerializer(clothe)
        return Response(serializer.data)


    def update(self, request, pk=None):
        clothe = Clothe.objects.get(id=pk)
        serializer = ClotheSerializer(instance=clothe, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(status=status.HTTP_202_ACCEPTED)

    def destroy(self, request, pk=None):
        clothe = Clothe.objects.get(id=pk)
        clothe.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)