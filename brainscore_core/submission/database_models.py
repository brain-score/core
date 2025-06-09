from peewee import Proxy, Model as PeeweeModel, CharField, ForeignKeyField, IntegerField, BooleanField, DateTimeField, \
    FloatField, TextField, PrimaryKeyField

database_proxy = Proxy()


class PeeweeBase(PeeweeModel):
    @classmethod
    def set_schema(cls, schema_name):
        for model in cls.__subclasses__():
            model._meta.schema = schema_name

    class Meta:
        database = database_proxy


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


class Reference(PeeweeBase):
    author = CharField()
    bibtex = TextField()
    url = CharField()
    year = IntegerField()

    class Meta:
        table_name = 'brainscore_reference'


class BenchmarkType(PeeweeBase):
    identifier = CharField(primary_key=True)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference, null=True)
    order = IntegerField()
    parent = ForeignKeyField(column_name='parent_id', field='identifier', model='self', null=True)
    visible = BooleanField(default=False, null=False)
    domain = CharField(max_length=200, default=None)
    owner = ForeignKeyField(column_name='owner_id', field='id', model=User, null=False)

    class Meta:
        table_name = 'brainscore_benchmarktype'


class BenchmarkMeta(PeeweeBase):
    identifier = CharField(primary_key=True)
    number_of_stimuli = IntegerField(null=True)
    number_of_recording_sites = IntegerField(null=True)
    recording_sites = CharField(max_length=100, null=True)
    behavioral_task = CharField(max_length=100, null=True)

    class Meta:
        table_name = 'brainscore_benchmarkmeta'


class Submission(PeeweeBase):
    jenkins_id = IntegerField()
    submitter = ForeignKeyField(column_name='submitter_id', field='id', model=User)
    timestamp = DateTimeField(null=True)
    model_type = CharField()
    status = CharField()

    class Meta:
        table_name = 'brainscore_submission'


class Model(PeeweeBase):
    name = CharField()
    owner = ForeignKeyField(column_name='owner_id', field='id', model=User)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference, null=True)
    submission = ForeignKeyField(column_name='submission_id', field='id', model=Submission)
    domain = CharField(max_length=200, default=None)
    visual_degrees = IntegerField(null=True)  # null during creation of new model without having model object loaded
    public = BooleanField()
    competition = CharField(max_length=200, default=None, null=True)

    class Meta:
        table_name = 'brainscore_model'


class ModelMeta(PeeweeBase):
    model = ForeignKeyField(column_name='model_id', field='id', model=Model, primary_key=True)
    architecture = CharField(max_length=100, null=True, default=None)
    model_family = CharField(max_length=100, null=True, default=None)
    total_parameter_count = IntegerField(null=True, default=None)
    total_layers = IntegerField(null=True, default=None)
    training_dataset = CharField(max_length=100, null=True, default=None)
    task_specialization = CharField(max_length=100, null=True, default=None)
    brainscore_link = CharField(max_length=256, null=True, default=None)
    hugging_face_link = CharField(max_length=256, null=True, default=None)
    trainable_parameter_count = IntegerField(null=True, default=None)
    trainable_layers = IntegerField(null=True, default=None)
    model_size_mb = FloatField(null=True, default=None)
    extra_notes = CharField(max_length=1000, null=True, default=None)
    runnable = BooleanField(default=True, null=True)

    class Meta:
        table_name = 'brainscore_modelmeta'


class BenchmarkStimuliMeta(PeeweeBase):
    id = PrimaryKeyField()
    num_stimuli = IntegerField(null=True)
    datatype = CharField(max_length=100, null=True)
    stimuli_subtype = CharField(max_length=100, null=True)
    total_size_mb = FloatField(null=True)
    brainscore_link = CharField(max_length=200, null=True)
    extra_notes = CharField(max_length=1000, null=True)

    class Meta:
        table_name = 'brainscore_benchmark_stimuli_meta'


class BenchmarkDataMeta(PeeweeBase):
    id = PrimaryKeyField()
    benchmark_type = CharField(max_length=100, null=True)
    task = CharField(max_length=100, null=True)
    region = CharField(max_length=100, null=True)
    hemisphere = CharField(max_length=100, null=True)
    num_recording_sites = IntegerField(null=True)
    duration_ms = FloatField(null=True)
    species = CharField(max_length=100, null=True)
    datatype = CharField(max_length=100, null=True)
    num_subjects = IntegerField(null=True)
    pre_processing = CharField(max_length=100, null=True)
    brainscore_link = CharField(max_length=200, null=True)
    extra_notes = CharField(max_length=1000, null=True)
    data_publicly_available = BooleanField(default=True, null=False)

    class Meta:
        table_name = 'brainscore_benchmark_data_meta'


class BenchmarkMetricMeta(PeeweeBase):
    id = PrimaryKeyField()
    type = CharField(max_length=100, null=True)
    reference = CharField(max_length=100, null=True)
    public = BooleanField(null=True)
    brainscore_link = CharField(max_length=200, null=True)
    extra_notes = CharField(max_length=1000, null=True)

    class Meta:
        table_name = 'brainscore_benchmark_metric_meta'


class BenchmarkInstance(PeeweeBase):
    benchmark = ForeignKeyField(column_name='benchmark_type_id', field='identifier', model=BenchmarkType)
    ceiling = FloatField(null=True)
    ceiling_error = FloatField(null=True)
    version = IntegerField(null=True)
    meta = ForeignKeyField(column_name='meta_id', field='id', model=BenchmarkMeta, null=True)
    data_meta = ForeignKeyField(column_name='data_meta_id', field='id', model=BenchmarkDataMeta, null=True)
    metric_meta = ForeignKeyField(column_name='metric_meta_id', field='id', model=BenchmarkMetricMeta, null=True)
    stimuli_meta = ForeignKeyField(column_name='stimuli_meta_id', field='id', model=BenchmarkStimuliMeta, null=True)

    class Meta:
        table_name = 'brainscore_benchmarkinstance'


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


def create_schema(schema_name):
    """
    Creates an isolated schema for testing purposes.
    """
    try:
        database_proxy.execute_sql(f'CREATE SCHEMA IF NOT EXISTS {schema_name}')
        print(f"Schema {schema_name} created successfully.")
    except Exception as e:
        print(f"Error creating schema {schema_name}: {e}")


def drop_schema(schema_name):
    """
    Drops a schema that was used for testing purposes.
    """
    database_proxy.execute_sql(f'DROP SCHEMA IF EXISTS {schema_name} CASCADE')


def create_tables():
    """
    Creates the necessary tables for a test database schema.
    """
    with database_proxy:
        database_proxy.create_tables(PeeweeBase.__subclasses__(), safe=True)
