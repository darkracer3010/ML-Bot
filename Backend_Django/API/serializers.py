from rest_framework import serializers
from .models import datasetItem

class datasetItemSerializer(serializers.ModelSerializer):
    class Meta: 
        model=datasetItem
        fields= ["acc","pv"]