"""Validate PR metadata as data, using trusted installed code and GitHub reads."""

import argparse
import base64
import json
import os
from urllib.request import Request, urlopen
from urllib.parse import quote
from .contract import load, MetadataError, MAX_BYTES
from .policy import protected_changes


def api(path):
    request = Request(
        "https://api.github.com" + path,
        headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def pages(path):
    result = []
    for page in range(1, 31):
        values = api(path + f"?per_page=100&page={page}")
        result.extend(values)
        if len(values) < 100:
            return result
    raise MetadataError("PR is too large for metadata validation")


def content(repository, path, ref):
    value = api(
        f"/repos/{repository}/contents/{quote(path, safe='/')}?ref={quote(ref, safe='')}"
    )
    if (
        value.get("type") != "file"
        or value.get("size", 0) > MAX_BYTES
        or value.get("encoding") != "base64"
    ):
        raise MetadataError("Invalid metadata file")
    return base64.b64decode(value["content"]).decode()


def check_pr(repository, number, domain, root, head_sha=None):
    from .contract import read_yaml

    pr = api(f"/repos/{repository}/pulls/{number}")
    if pr["base"]["repo"]["full_name"] != repository:
        raise MetadataError("Repository mismatch")
    if head_sha and pr["head"]["sha"] != head_sha:
        raise MetadataError(
            "The PR head changed; rerun validation for the latest revision"
        )
    findings = []
    for item in pages(f"/repos/{repository}/pulls/{number}/files"):
        path = item["filename"]
        previous = item.get("previous_filename", "")

        def relevant(candidate):
            return candidate.startswith(root.rstrip("/") + "/") and candidate.rsplit(
                "/", 1
            )[-1] in {"metadata.yaml", "metadata.yml"}

        if not relevant(path) and not relevant(previous):
            continue
        if item["status"] in {"renamed", "removed"}:
            raise MetadataError("Metadata removal/rename requires a separate migration")
        if len(path[len(root.rstrip("/") + "/") :].split("/")) != 2:
            raise MetadataError(
                "Metadata must be inside one registered plugin directory"
            )
        raw = content(pr["head"]["repo"]["full_name"], path, pr["head"]["sha"])
        header = read_yaml(raw)
        old_raw = (
            content(repository, path, pr["base"]["sha"])
            if item["status"] != "added"
            else None
        )
        old_header = read_yaml(old_raw) if old_raw is not None else {}
        was_v2 = (
            isinstance(old_header, dict) and old_header.get("schema_version") == "2.0"
        )
        if not isinstance(header, dict) or header.get("schema_version") != "2.0":
            if was_v2:
                raise MetadataError("Metadata cannot be downgraded to a legacy schema")
            continue
        after = load(raw, domain)
        if item["status"] == "added":
            findings.append(
                {
                    "path": path,
                    "migration": "initial v2 publication requires a merged PR",
                }
            )
        if item["status"] != "added":
            if was_v2:
                before = load(old_raw, domain)
                if set(before["models"]) != set(after["models"]):
                    raise MetadataError(
                        "Model additions/removals require a registration migration"
                    )
                for key, entry in before["models"].items():
                    changed = protected_changes(entry, after["models"][key])
                    if changed:
                        findings.append(
                            {"path": path, "model": key, "protected_fields": changed}
                        )
            else:
                findings.append(
                    {
                        "path": path,
                        "migration": "legacy-to-v2 conversion requires a merged PR",
                    }
                )
        findings.append(
            {"path": path, "models": list(after["models"]), "status": "valid"}
        )
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head-sha")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()
    findings = check_pr(
        args.repository, args.pr, args.domain, args.model_root, args.head_sha
    )
    print(json.dumps(findings, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write("Model metadata v2 validation passed.\n\n")
            stream.write("Open-PR data has not been published to the live database.\n")


if __name__ == "__main__":
    main()
