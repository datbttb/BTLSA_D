# Generated by Django 4.1.7 on 2023-05-10 15:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user_model', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user_registration',
            name='staff',
            field=models.IntegerField(default=3),
        ),
    ]
