from django.db import models

# Create your models here.
class Question(models.Model):
    question_text = models.charField(max_length=100)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

class Choice(models.Model):
    question = models.ForeignKey(Question,on_delete=models.CASCADE)
    choice_text = models.charField(max_length=100)
    votes = models.IntegrField(default=0)

    def __str__(self):
        return self.choice_text