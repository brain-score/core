"""Lossless conversion into the website's six metadata tables."""

import hashlib
from .contract import FIELD_SPECS, get_path, put_path, validate


def to_tables(document):
    validate(document)
    tables = {
        key: []
        for key in (
            "models",
            "model_datasets",
            "intended_use",
            "contributors",
            "model_relationships",
            "assertions",
        )
    }
    for identifier, entry in document["models"].items():
        key = {"domain": document["domain"], "identifier": identifier}
        tables["models"].append(
            dict(
                key,
                **{
                    column: get_path(entry, path)
                    for path, (column, _, _) in FIELD_SPECS.items()
                },
            )
        )
        ordinal = 0
        for path, role in (
            ("/data/training_datasets", None),
            ("/eval/test_datasets", "test"),
            ("/eval/validation_datasets", "validation"),
        ):
            for dataset in get_path(entry, path) or []:
                tables["model_datasets"].append(
                    dict(
                        key,
                        ordinal=ordinal,
                        dataset_identifier=dataset.get("identifier"),
                        dataset_name=dataset["name"],
                        role=role or dataset["role"],
                        description=dataset.get("description"),
                    )
                )
                ordinal += 1
        for category in ("applications", "users", "limitations", "biases"):
            for ordinal, value in enumerate(get_path(entry, "/use/" + category) or []):
                tables["intended_use"].append(
                    dict(key, category=category, ordinal=ordinal, value=value)
                )
        for kind in ("creators", "organizations"):
            for ordinal, value in enumerate(get_path(entry, "/people/" + kind) or []):
                tables["contributors"].append(
                    dict(key, kind=kind, ordinal=ordinal, name=value)
                )
        for ordinal, base in enumerate(get_path(entry, "/lineage/base_models") or []):
            tables["model_relationships"].append(
                dict(
                    key,
                    ordinal=ordinal,
                    base_identifier=base.get("identifier"),
                    base_name=base["name"],
                    relationship=base["relationship"],
                )
            )
        for assertion in entry.get("assertions", []):
            sources = [entry["sources"][ref] for ref in assertion.get("sources", [])]
            source = "; ".join(
                s.get("url") or s.get("citation") or "Source review needed"
                for s in sources
            )
            tables["assertions"].append(
                dict(
                    key,
                    path=assertion["path"],
                    status=assertion["status"],
                    source=source or None,
                )
            )
    return tables


def from_tables(tables, domain):
    document = {"schema_version": "2.0", "domain": domain, "models": {}}
    for row in tables["models"]:
        if row["domain"] != domain:
            continue
        entry = {"sources": {}, "assertions": []}
        for path, (column, _, _) in FIELD_SPECS.items():
            put_path(entry, path, row.get(column))
        document["models"][row["identifier"]] = entry
    for name, rows in tables.items():
        if name == "models":
            continue
        for row in rows:
            if row["domain"] != domain or row["identifier"] not in document["models"]:
                continue
            entry = document["models"][row["identifier"]]
            if name == "model_datasets":
                path = {
                    "test": "/eval/test_datasets",
                    "validation": "/eval/validation_datasets",
                }.get(row["role"], "/data/training_datasets")
                value = {
                    "identifier": row["dataset_identifier"],
                    "name": row["dataset_name"],
                    "description": row["description"],
                }
                if path.startswith("/data/"):
                    value["role"] = row["role"]
            elif name == "intended_use":
                path, value = "/use/" + row["category"], row["value"]
            elif name == "contributors":
                path, value = "/people/" + row["kind"], row["name"]
            elif name == "model_relationships":
                path = "/lineage/base_models"
                value = {
                    "identifier": row["base_identifier"],
                    "name": row["base_name"],
                    "relationship": row["relationship"],
                }
            else:
                # Preserve evidence at known v2 paths; report unsupported paths to
                # the caller instead of silently dropping legacy assertions.
                path = row["path"]
                if path == "/data/dataset_size":
                    path = "/data/summary"
                refs = []
                if row.get("source"):
                    raw_source = row["source"]
                    source_id = (
                        "curation_workbook"
                        if raw_source == "curation_workbook"
                        else "legacy_"
                        + hashlib.sha256(raw_source.encode()).hexdigest()[:16]
                    )
                    entry["sources"][source_id] = {
                        "kind": "unreviewed",
                        "citation": raw_source,
                    }
                    refs = [source_id]
                entry["assertions"].append(
                    {"path": path, "status": row["status"], "sources": refs}
                )
                continue
            values = get_path(entry, path) or []
            values.append(value)
            put_path(entry, path, values)
    return validate(document, domain)


def legacy_projection(entry):
    """Project v2 facts into legacy fields without inventing numeric values."""
    result = dict(entry.get("legacy", {}))
    count = get_path(entry, "/model/parameter_count")
    result["total_parameter_count"] = count
    family = get_path(entry, "/model/architecture/family")
    architecture = {
        "convolutional_neural_network": "DCNN",
        "vision_transformer": "Transformer",
        "recurrent_convolutional_neural_network": "Recurrent",
        "hybrid_convolutional_transformer": "Hybrid",
        "raw_pixels": "Pixels",
    }.get(family, family)
    result["architecture"] = architecture
    layers = get_path(entry, "/model/trainable_layers")
    result["trainable_layers"] = (
        int(layers)
        if isinstance(layers, str) and layers.isdigit() and int(layers) <= 2147483647
        else None
    )
    return result


def from_legacy(values):
    """Preserve legacy siblings during file conversion; never claim verification."""
    entry = {
        "legacy": dict(values),
        "sources": {
            "legacy": {
                "kind": "unreviewed",
                "citation": "Existing plugin metadata; primary sources need review.",
            }
        },
        "assertions": [],
    }
    mapping = {
        "total_parameter_count": "/model/parameter_count",
        "trainable_layers": "/model/trainable_layers",
        "training_dataset": "/data/summary",
        "brainscore_link": "/provenance/source_url",
    }
    for key, path in mapping.items():
        value = values.get(key)
        if value is not None:
            if path == "/model/trainable_layers":
                value = str(value)
            put_path(entry, path, value)
            entry["assertions"].append(
                {"path": path, "status": "probable", "sources": ["legacy"]}
            )
    architecture = values.get("architecture")
    if architecture:
        family = {
            "DCNN": "convolutional_neural_network",
            "Transformer": "vision_transformer",
        }.get(architecture, architecture)
        put_path(entry, "/model/architecture/family", family)
        entry["assertions"].append(
            {"path": "/model/architecture", "status": "probable", "sources": ["legacy"]}
        )
    return entry
