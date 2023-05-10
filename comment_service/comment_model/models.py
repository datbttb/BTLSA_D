from django.db import models

class Comment(models.Model):
    id = models.AutoField(primary_key=True)
    message = models.TextField(null=True)
    publisher = models.IntegerField(null=False)
    product = models.IntegerField(null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'commentservice'
        ordering = ['-created_at']

    def __unicode__(self):
        return self.name
