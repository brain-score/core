import boto3
import logging

_logger = logging.getLogger(__name__)


class UniqueKeyDict(dict):
    def __init__(self, reload=False, **kwargs):
        super().__init__(**kwargs)
        self.reload = reload

    def __setitem__(self, key, *args, **kwargs):
        if key in self:
            raise KeyError("Key '{}' already exists with value '{}'.".format(key, self[key]))
        super(UniqueKeyDict, self).__setitem__(key, *args, **kwargs)

    def __getitem__(self, item):
        value = super(UniqueKeyDict, self).__getitem__(item)
        if self.reload and hasattr(value, 'reload'):
            _logger.warning(f'{item} is accessed again and reloaded')
            value.reload()
        return value


def get_secret(secret_name: str, region_name: str = 'us-east-2') -> str:
    session = boto3.session.Session()
    _logger.debug("Fetch secret from secret manager")
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
    )
    secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    _logger.debug('Secret successfully fetched')
    if 'SecretString' in secret_value_response:
        return secret_value_response['SecretString']
    else:
        return secret_value_response['SecretBinary']
