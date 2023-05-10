# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from product_model.models import product_details
from rest_framework import viewsets, status
from product_model.serializers import ProductSerializer
from rest_framework.response import Response


class ProductViewSet(viewsets.ViewSet):
    def list(self, request):
        products = product_details.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)