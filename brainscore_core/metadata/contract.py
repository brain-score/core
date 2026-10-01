import math
from urllib.parse import urlparse
import yaml
from yaml.tokens import AliasToken, AnchorToken

MAX_BYTES = 512_000
STATUSES = {"verified", "probable", "uncertain", "undocumented"}
SOURCE_KINDS = {"paper", "huggingface", "other", "unreviewed"}
# path -> storage column, value type, maximum text length (None is unbounded)
FIELD_SPECS = {
    "/model/display_name": ("display_name", str, 200),
    "/model/version": ("version", str, 200),
    "/model/architecture/family": ("architecture_family", str, 100),
    "/model/architecture/description": ("architecture_description", str, None),
    "/model/parameter_count": ("parameter_count", int, None),
    "/model/parameter_count_exact": ("parameter_count_exact", bool, None),
    "/model/trainable_layers": ("trainable_layers", str, None),
    "/model/recurrent": ("recurrent", bool, None),
    "/model/input_resolution/channels": ("input_channels", int, None),
    "/model/input_resolution/height": ("input_height", int, None),
    "/model/input_resolution/width": ("input_width", int, None),
    "/model/visual_degrees": ("visual_degrees", float, None),
    "/model/visual_degrees_description": ("visual_degrees_description", str, None),
    "/model/supervision/type": ("supervision_type", str, 100),
    "/model/supervision/description": ("supervision_description", str, None),
    "/training/process": ("training_process", str, None),
    "/training/objective": ("training_objective", str, None),
    "/training/loss": ("loss_function", str, None),
    "/training/learning_rate": ("learning_rate", str, None),
    "/training/batch_size": ("batch_size", str, None),
    "/training/preprocessing": ("preprocessing_description", str, None),
    "/data/summary": ("dataset_summary", str, None),
    "/io/modality": ("input_modality", str, 50),
    "/io/interface": ("interface_description", str, None),
    "/io/input_format": ("input_format", str, None),
    "/io/output_format": ("output_format", str, None),
    "/io/tokenizer": ("tokenizer", str, None),
    "/provenance/weights_provider": ("weights_provider", str, 500),
    "/provenance/checkpoint": ("checkpoint_identifier", str, 500),
    "/provenance/source_url": ("source_url", str, 1000),
    "/provenance/curation_confidence": ("curation_confidence", str, 50),
    "/legal/license": ("license", str, 500),
}
LIST_PATHS = (
    "/data/training_datasets",
    "/eval/test_datasets",
    "/eval/validation_datasets",
    "/people/creators",
    "/people/organizations",
    "/use/applications",
    "/use/users",
    "/use/limitations",
    "/use/biases",
    "/lineage/base_models",
)
LEGACY_KEYS = {
    "architecture",
    "model_family",
    "total_parameter_count",
    "trainable_parameter_count",
    "total_layers",
    "trainable_layers",
    "model_size_mb",
    "training_dataset",
    "task_specialization",
    "brainscore_link",
    "huggingface_link",
    "extra_notes",
    "runnable",
}


class MetadataError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    pairs = loader.construct_pairs(node, deep=deep)
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise MetadataError("Mapping keys must be unique strings")
        result[key] = value
    return result


UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def get_path(entry, path):
    node = entry
    for key in path.strip("/").split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def put_path(entry, path, value):
    keys = path.strip("/").split("/")
    for key in keys[:-1]:
        entry = entry.setdefault(key, {})
    entry[keys[-1]] = value


def mapping(value, allowed, where):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise MetadataError(f"{where}: expected a mapping with known keys")


def text(value, where, required=False, limit=10000):
    if value is None and not required:
        return
    if (
        not isinstance(value, str)
        or len(value) > limit
        or (required and not value.strip())
    ):
        raise MetadataError(
            f"{where}: expected {'nonempty ' if required else ''}text, maximum {limit} characters"
        )


def source_kind(source):
    host = (urlparse(source.get("url") or "").hostname or "").lower()
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        return "huggingface"
    if host in {
        "doi.org",
        "dx.doi.org",
        "arxiv.org",
        "www.arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
    }:
        return "paper"
    return source["kind"]


def validate(document, domain=None):
    mapping(document, {"schema_version", "domain", "models"}, "document")
    if document.get("schema_version") != "2.0":
        raise MetadataError('schema_version must be the string "2.0"')
    text(document.get("domain"), "domain", required=True, limit=200)
    if domain is not None and document["domain"] != domain:
        raise MetadataError("Document domain does not match its repository")
    models = document.get("models")
    if not isinstance(models, dict) or not 1 <= len(models) <= 100:
        raise MetadataError("models must contain between 1 and 100 entries")
    seen = set()
    all_paths = set(FIELD_SPECS) | set(LIST_PATHS)
    groups = {p.rsplit("/", 1)[0] for p in all_paths} | {
        "/model",
        "/training",
        "/data",
        "/eval",
        "/io",
        "/provenance",
        "/legal",
        "/people",
        "/use",
        "/lineage",
    }
    for identifier, entry in models.items():
        text(identifier, "model identifier", required=True, limit=200)
        if identifier.lower() in seen:
            raise MetadataError("Model identifiers must be unique ignoring case")
        seen.add(identifier.lower())
        mapping(
            entry,
            {p.split("/")[1] for p in all_paths} | {"sources", "assertions", "legacy"},
            identifier,
        )

        def walk(node, path=""):
            for key, value in node.items():
                here = path + "/" + key
                if path == "" and key in {"sources", "assertions", "legacy"}:
                    continue
                if here in FIELD_SPECS:
                    _, kind, maximum = FIELD_SPECS[here]
                    if value is None:
                        continue
                    if kind == str:
                        text(value, here, limit=maximum or 10000)
                    elif kind == float:
                        if (
                            type(value) not in (float, int)
                            or not math.isfinite(value)
                            or value < 0
                        ):
                            raise MetadataError(
                                here + ": expected a finite nonnegative number"
                            )
                    elif type(value) != kind or (
                        kind == int and not 0 <= value <= 9223372036854775807
                    ):
                        raise MetadataError(here + ": invalid value type or range")
                    if (
                        here.endswith(("/channels", "/height", "/width"))
                        and value > 2147483647
                    ):
                        raise MetadataError(here + ": integer exceeds storage range")
                elif here in LIST_PATHS:
                    if not isinstance(value, list) or len(value) > 500:
                        raise MetadataError(
                            here + ": expected a list of at most 500 entries"
                        )
                    for item in value:
                        if here.startswith(("/people/", "/use/")):
                            text(item, here, required=True)
                        elif here == "/lineage/base_models":
                            mapping(item, {"identifier", "name", "relationship"}, here)
                            text(item.get("identifier"), here, limit=200)
                            text(item.get("name"), here, required=True, limit=200)
                            if not isinstance(
                                item.get("relationship"), str
                            ) or item.get("relationship") not in {
                                "variant_of",
                                "fine_tuned_from",
                                "derived_from",
                            }:
                                raise MetadataError(here + ": invalid relationship")
                            if (
                                item.get("identifier") or ""
                            ).lower() == identifier.lower():
                                raise MetadataError(
                                    here + ": a model cannot be its own base"
                                )
                        else:
                            allowed = {"identifier", "name", "description"} | (
                                {"role"} if here.startswith("/data/") else set()
                            )
                            mapping(item, allowed, here)
                            text(item.get("identifier"), here, limit=200)
                            text(item.get("name"), here, required=True)
                            text(item.get("description"), here)
                            if here.startswith("/data/") and (
                                not isinstance(item.get("role"), str)
                                or item.get("role")
                                not in {"training", "pretraining", "fine_tuning"}
                            ):
                                raise MetadataError(here + ": invalid dataset role")
                elif here in groups and isinstance(value, dict):
                    walk(value, here)
                else:
                    raise MetadataError(here + ": unknown field or invalid structure")

        walk(entry)
        sources = entry.get("sources", {})
        if not isinstance(sources, dict) or len(sources) > 200:
            raise MetadataError("sources must be a mapping of at most 200 references")
        for key, source in sources.items():
            text(key, "source ID", required=True, limit=100)
            mapping(source, {"kind", "url", "citation"}, "source")
            if (
                not isinstance(source.get("kind"), str)
                or source.get("kind") not in SOURCE_KINDS
            ):
                raise MetadataError("Unknown source kind")
            text(source.get("citation"), "citation")
            text(source.get("url"), "source URL", limit=2000)
            url = source.get("url")
            try:
                parsed = urlparse(url or "")
                invalid = url and (
                    parsed.scheme != "https" or not parsed.hostname or parsed.username
                )
            except ValueError as exc:
                raise MetadataError("Invalid source URL") from exc
            if invalid:
                raise MetadataError(
                    "Source URLs must be absolute HTTPS URLs without credentials"
                )
            if source["kind"] != "unreviewed" and not (url or source.get("citation")):
                raise MetadataError("Classified sources need a citation or URL")
            if source_kind(source) != source["kind"]:
                raise MetadataError("Source kind contradicts its URL")
        assertions = entry.get("assertions", [])
        if not isinstance(assertions, list) or len(assertions) > 200:
            raise MetadataError("assertions must be a list of at most 200 entries")
        paths = set()
        for assertion in assertions:
            mapping(assertion, {"path", "status", "sources"}, "assertion")
            path = assertion.get("path")
            if (
                not isinstance(path, str)
                or path not in all_paths | groups
                or path in paths
                or len(path) > 200
            ):
                raise MetadataError(
                    "Assertion path must be a unique supported field or section"
                )
            paths.add(path)
            if (
                not isinstance(assertion.get("status"), str)
                or assertion.get("status") not in STATUSES
            ):
                raise MetadataError("Unknown assertion status")
            refs = assertion.get("sources", [])
            if (
                not isinstance(refs, list)
                or any(not isinstance(ref, str) or ref not in sources for ref in refs)
                or len(refs) != len(set(refs))
            ):
                raise MetadataError(
                    "Assertion source references must exist and be unique"
                )
        legacy = entry.get("legacy", {})
        mapping(legacy, LEGACY_KEYS, "legacy")
        for key, value in legacy.items():
            if value is None:
                continue
            if key in {
                "total_parameter_count",
                "trainable_parameter_count",
                "total_layers",
                "trainable_layers",
            }:
                maximum = (
                    9223372036854775807
                    if key.endswith("parameter_count")
                    else 2147483647
                )
                if type(value) != int or not 0 <= value <= maximum:
                    raise MetadataError(
                        "Legacy counts must be nonnegative integers within storage range"
                    )
            elif key == "model_size_mb":
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise MetadataError(
                        "Legacy model size must be a finite nonnegative number"
                    )
            elif key == "runnable":
                if type(value) != bool:
                    raise MetadataError("Legacy runnable must be a boolean")
            else:
                text(
                    value,
                    "legacy/" + key,
                    limit=1000
                    if key == "extra_notes"
                    else 256
                    if key.endswith("_link")
                    else 100,
                )
    return document


def read_yaml(content):
    if (
        len(content.encode("utf-8") if isinstance(content, str) else content)
        > MAX_BYTES
    ):
        raise MetadataError("Metadata file exceeds 512 KB")
    try:
        if any(
            isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(content)
        ):
            raise MetadataError("YAML anchors and aliases are not supported")
        return yaml.load(content, Loader=UniqueLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise MetadataError("Invalid YAML") from exc


def load(content, domain=None):
    return validate(read_yaml(content), domain)


def dump(document):
    validate(document)
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
