from django.urls import path
from .  import views

urlpatterns=[
    path('predict',views.predict),
    path('getData',views.getData),
    
]