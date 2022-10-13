from datetime import datetime

from brainscore_core.submission.database_models import Score, Model, Submission, BenchmarkInstance, BenchmarkType, \
    Reference, User


def clear_schema():
    Score.delete().execute()
    Model.delete().execute()
    Submission.delete().execute()
    BenchmarkInstance.delete().execute()
    BenchmarkType.delete().execute()
    Reference.delete().execute()
    User.delete().execute()


def init_users():
    User.create(id=1, email='test@brainscore.com', is_active=True, is_staff=False, is_superuser=False,
                last_login=datetime.now(), password='abcde')
    User.create(id=2, email='admin@brainscore.com', is_active=True, is_staff=True, is_superuser=True,
                last_login=datetime.now(), password='abcdef')
