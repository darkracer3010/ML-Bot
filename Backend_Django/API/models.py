from django.db import models

class datasetItem(models.Model):
    desc=models.CharField(max_length=100)
    name=models.CharField(max_length=20)
    atc=models.IntegerField()
    dataset=models.CharField(max_length=50)
    acc=models.IntegerField()
    pv=models.IntegerField()
    

    def __str__(self):
        return self.name
