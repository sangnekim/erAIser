from django.db import models
from django.db.models.base import Model

# Create your models here.
class ImageTarget(models.Model):
    username = models.CharField(max_length=10)
    photo = models.ImageField(upload_to="", null=True)


class Video(models.Model):
    title = models.CharField(max_length = 500)
    videofile = models.FileField(upload_to = "./video", null = True)

    class Meta:
        verbose_name = 'video'
        verbose_name_plural = 'videos'

    def __str__(self):
        return self.title

class Image(models.Model):
    title = models.CharField(max_length = 500)
    imagefile = models.FileField(upload_to = "./image", null = True)

    class Meta:
        verbose_name = 'image'
        verbose_name_plural = 'images'

    def __str__(self):
        return self.title