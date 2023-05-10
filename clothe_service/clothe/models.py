from django.db import models

class Clothe(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=9, decimal_places=2)
    quantity = models.IntegerField()
    publisher = models.TextField(null=True)
    loai = models.TextField(max_length=255, default="clothe")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'clothe'
        ordering = ['-created_at']

    def __unicode__(self):
        return self.name