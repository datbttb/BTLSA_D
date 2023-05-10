from django.db import models

class Image(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to='image/', null=True)
    product= models.IntegerField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'images'
        ordering = ['-created_at']

    def __unicode__(self):
        return self.name
