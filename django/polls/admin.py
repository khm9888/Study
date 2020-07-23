import os
import sys
sys.path.append("D:\Study\django")

from django.contrib import admin

# Register your models here.

from polls.models import Question, Choice

admin.site.register(Question)
admin.site.register(Choice)
