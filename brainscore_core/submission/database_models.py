from peewee import Proxy, Model as PeeweeModel, CharField, ForeignKeyField, IntegerField, BooleanField, DateTimeField, \
    FloatField, TextField, PrimaryKeyField

database_proxy = Proxy()


class PeeweeBase(PeeweeModel):
    class Meta:
        database = database_proxy


class Reference(PeeweeBase):
    author = CharField()
    bibtex = TextField()
    url = CharField()
    year = IntegerField()

    class Meta:
        table_name = 'brainscore_reference'
        schema = 'public'


class BenchmarkType(PeeweeBase):
    identifier = CharField(primary_key=True)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference)
    order = IntegerField()
    parent = ForeignKeyField(column_name='parent_id', field='identifier', model='self', null=True)
    visible = BooleanField(default=False, null=False)

    class Meta:
        table_name = 'brainscore_benchmarktype'
        schema = 'public'


class BenchmarkMeta(PeeweeBase):
    number_of_images = IntegerField(null=True)
    number_of_recording_sites = IntegerField(null=True)
    recording_sites = CharField(max_length=100, null=True)
    behavioral_task = CharField(max_length=100, null=True)

    class Meta:
        table_name = 'brainscore_benchmarkmeta'
        schema = 'public'


class BenchmarkInstance(PeeweeBase):
    benchmark = ForeignKeyField(column_name='benchmark_type_id', field='identifier', model=BenchmarkType)
    ceiling = FloatField(null=True)
    ceiling_error = FloatField(null=True)
    version = IntegerField(null=True)
    meta = ForeignKeyField(model=BenchmarkMeta)

    class Meta:
        table_name = 'brainscore_benchmarkinstance'
        schema = 'public'


class User(PeeweeBase):
    id = PrimaryKeyField()
    email = CharField(index=True, null=True)
    is_active = BooleanField()
    is_staff = BooleanField()
    is_superuser = BooleanField()
    last_login = DateTimeField(null=True)
    password = CharField()

    class Meta:
        table_name = 'brainscore_user'
        schema = 'public'


class Submission(PeeweeBase):
    jenkins_id = IntegerField() 
    # IDs will not be incremental when resubmitting models
    submitter = ForeignKeyField(column_name='submitter_id', field='id', model=User)
    timestamp = DateTimeField(null=True)
    model_type = CharField()
    status = CharField()

    class Meta:
        table_name = 'brainscore_submission'
        schema = 'public'


class Model(PeeweeBase):
    name = CharField()
    owner = ForeignKeyField(column_name='owner_id', field='id', model=User)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference)
    submission = ForeignKeyField(column_name='submission_id', field='id', model=Submission)
    domain = CharField(max_length=200, default=None, null=True)
    visual_degrees = IntegerField(null=True)  # null during creation of new model without having model object loaded
    public = BooleanField()
    competition = CharField(max_length=200, default=None, null=True)

    class Meta:
        table_name = 'brainscore_model'
        schema = 'public'


class Score(PeeweeBase):
    benchmark = ForeignKeyField(column_name='benchmark_id', field='id', model=BenchmarkInstance)
    end_timestamp = DateTimeField(null=True)
    error = FloatField(null=True)
    model = ForeignKeyField(column_name='model_id', field='id', model=Model)
    score_ceiled = FloatField(null=True)
    score_raw = FloatField(null=True)
    start_timestamp = DateTimeField(null=True)
    comment = CharField(null=True, max_length=1_000)

    class Meta:
        table_name = 'brainscore_score'
        schema = 'public'


def clear_schema():
    """
    Delete the contents of all tables.
    This function is meant for testing only, use with caution.
    """
    Score.delete().execute()
    Model.delete().execute()
    Submission.delete().execute()
    BenchmarkInstance.delete().execute()
    BenchmarkType.delete().execute()
    Reference.delete().execute()
    User.delete().execute()
