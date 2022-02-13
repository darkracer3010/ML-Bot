from glob import glob
from itsdangerous import Serializer
from rest_framework.response import Response
from rest_framework import viewsets
from django.shortcuts import render
from .serializers import datasetItemSerializer
from .models import datasetItem
from rest_framework.decorators import api_view
from .algorithms import getModel
from rest_framework import status
import json

model=None

@api_view(["POST"])
def predict(request):
    global model
    req=request.POST
    # #result={"name","model_object","accuracy"}
    result=getModel(req["url"][0])
    if model==None:
        model=result["model"]
    return Response(result["name"] + " has good accuracy in given dataset")

@api_view(["GET"])
def getData(request):
    req=request.GET
    global model
    if model==None:
        return Response({"Problem" : "Please upload dataset 1st"}, status=status.HTTP_303_SEE_OTHER)#sheesh
    result=model.predict(req["0"][0])
    dic={"acc":98.34,"value":1234}
    return Response(dic)
