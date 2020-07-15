from django.db import models

# Create your models here.

class Question(models.Model):
    question_text = models.CharField(max_length=2000)
    pub_data = models.DateTimeField('data published')

    def __str__(self):
        return self.question_text

class Choice(models.Model):
    question_text = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_test = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text