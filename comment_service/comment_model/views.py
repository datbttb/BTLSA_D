from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response

from comment_model.models import Comment
from comment_model.serializers import CommentSerializer


class CommentViewSet(viewsets.ViewSet):
    def list(self, request):
        comments = Comment.objects.all()
        serializer = CommentSerializer(comments, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer= CommentSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def retrieve(self, request, pk=None):
        comment = Comment.objects.get(id=pk)
        serializer = CommentSerializer(comment)
        return Response(serializer.data)


    def update(self, request, pk=None):
        comment = Comment.objects.get(id=pk)
        serializer = CommentSerializer(instance=comment, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(status=status.HTTP_202_ACCEPTED)

    def destroy(self, request, pk=None):
        comment = Comment.objects.get(id=pk)
        comment.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
