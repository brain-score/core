from datetime import datetime

from brainscore_core.submission.database_models import User


def init_users():
    User.create(id=1, email='test@brainscore.com', is_active=True, is_staff=False, is_superuser=False,
                last_login=datetime.now(), password='abcde')
    User.create(id=2, email='admin@brainscore.com', is_active=True, is_staff=True, is_superuser=True,
                last_login=datetime.now(), password='abcdef')
