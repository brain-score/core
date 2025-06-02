import logging
from peewee import DoesNotExist
from brainscore_core.submission.database_models import (
    BenchmarkInstance, BenchmarkStimuliMeta,
    BenchmarkDataMeta, BenchmarkMetricMeta
)

logger = logging.getLogger(__name__)


def get_existing_meta_ids(benchmark_instances):
    """Extract existing metadata IDs from benchmark instances."""
    if not benchmark_instances:
        return {
            'stimuli_meta_id': None,
            'data_meta_id': None,
            'metric_meta_id': None
        }

    instance = benchmark_instances[0]
    return {
        'stimuli_meta_id': instance.stimuli_meta_id if hasattr(instance, 'stimuli_meta_id') else None,
        'data_meta_id': instance.data_meta_id if hasattr(instance, 'data_meta_id') else None,
        'metric_meta_id': instance.metric_meta_id if hasattr(instance, 'metric_meta_id') else None
    }


def update_or_create_meta_record(model_class, existing_id, field_mapping):
    """Generic helper function to update or create metadata records."""
    if existing_id:
        try:
            record = model_class.get(model_class.id == existing_id)
            for field, value in field_mapping.items():
                setattr(record, field, value)
            record.save()
            logger.info(f"Updated existing {model_class.__name__} record id={existing_id}")
            return existing_id
        except model_class.DoesNotExist:
            logger.warning(f"{model_class.__name__} with id={existing_id} not found, creating new")

    # create new record
    record = model_class.create(**field_mapping)
    logger.info(f"Created new {model_class.__name__} record id={record.id}")
    return record.id


def process_stimuli_metadata(stimulus_data, existing_id):
    """Process stimulus metadata and update or create record."""
    field_mapping = {
        'num_stimuli': stimulus_data.get('num_stimuli'),
        'datatype': stimulus_data.get('datatype'),
        'stimuli_subtype': stimulus_data.get('stimuli_subtype'),
        'total_size_mb': stimulus_data.get('total_size_mb'),
        'brainscore_link': stimulus_data.get('brainscore_link'),
        'extra_notes': stimulus_data.get('extra_notes')
    }
    return update_or_create_meta_record(BenchmarkStimuliMeta, existing_id, field_mapping)


def process_data_metadata(data_info, existing_id):
    """Process data metadata and update or create record."""
    field_mapping = {
        'benchmark_type': data_info.get('benchmark_type'),
        'task': data_info.get('task'),
        'region': data_info.get('region'),
        'hemisphere': data_info.get('hemisphere'),
        'num_recording_sites': data_info.get('num_recording_sites'),
        'duration_ms': data_info.get('duration_ms'),
        'species': data_info.get('species'),
        'datatype': data_info.get('datatype'),
        'num_subjects': data_info.get('num_subjects'),
        'pre_processing': data_info.get('pre_processing'),
        'brainscore_link': data_info.get('brainscore_link'),
        'extra_notes': data_info.get('extra_notes'),
        'data_publicly_available': data_info.get('data_publicly_available'),
    }
    return update_or_create_meta_record(BenchmarkDataMeta, existing_id, field_mapping)


def process_metric_metadata(metric_info, existing_id):
    """Process metric metadata and update or create record."""
    public_value = False if metric_info.get('public') is None else metric_info.get('public')
    field_mapping = {
        'type': metric_info.get('type'),
        'reference': metric_info.get('reference'),
        'public': public_value,
        'brainscore_link': metric_info.get('brainscore_link'),
        'extra_notes': metric_info.get('extra_notes')
    }
    return update_or_create_meta_record(BenchmarkMetricMeta, existing_id, field_mapping)


def update_benchmark_instances(benchmark_instances, meta_ids):
    """Update benchmark instances with new metadata IDs."""
    for instance in benchmark_instances:
        update_fields = {}
        if 'stimuli_meta_id' in meta_ids and meta_ids['stimuli_meta_id'] is not None:
            update_fields['stimuli_meta'] = meta_ids['stimuli_meta_id']
        if 'data_meta_id' in meta_ids and meta_ids['data_meta_id'] is not None:
            update_fields['data_meta'] = meta_ids['data_meta_id']
        if 'metric_meta_id' in meta_ids and meta_ids['metric_meta_id'] is not None:
            update_fields['metric_meta'] = meta_ids['metric_meta_id']

        if update_fields:
            query = BenchmarkInstance.update(**update_fields).where(BenchmarkInstance.id == instance.id)
            query.execute()
            logger.info(f"Updated BenchmarkInstance {instance.id} with metadata IDs")


class BenchmarkMetaResult:
    """A class to hold benchmark metadata results and provide the expected interface."""

    def __init__(self, benchmark_identifier, **metadata_ids):
        self.identifier = benchmark_identifier
        self.id = metadata_ids.get('stimuli_meta_id')  # use one of the IDs as primary ID
        for key, value in metadata_ids.items():
            setattr(self, key, value)
