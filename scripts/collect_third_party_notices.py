#!/usr/bin/env python3
"""Collect licence texts and exact package manifests of a conda installation.

Run before ``conda clean``: conda keeps each package's licence files only in
``pkgs/<package>/info/licenses``, which cleaning deletes. The output lets a
distributed image carry every licence and answer source requests:

    <out>/licenses/conda/<name>-<version>/   licence files of each package
    <out>/manifests/conda-<env>.tsv          name, version, build, licence, url
    <out>/manifests/pip-<env>.tsv            pip-installed name, version, licence

Uses only the standard library, so the base Python of any conda works.
"""

import argparse
import csv
import json
import shutil
import sys
from email.parser import Parser
from pathlib import Path


def conda_envs(root: Path) -> list[tuple[str, Path]]:
    envs = [("base", root)]
    envs_dir = root / "envs"
    if envs_dir.is_dir():
        envs += [(p.name, p) for p in sorted(envs_dir.iterdir()) if (p / "conda-meta").is_dir()]
    return envs


def conda_records(prefix: Path) -> list[dict]:
    records = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            print(f"warning: unreadable conda record {path}", file=sys.stderr)
    return records


def license_dir(root: Path, record: dict) -> Path | None:
    candidates = []
    if record.get("extracted_package_dir"):
        candidates.append(Path(record["extracted_package_dir"]))
    fn = record.get("fn", "")
    for suffix in (".conda", ".tar.bz2"):
        if fn.endswith(suffix):
            candidates.append(root / "pkgs" / fn[: -len(suffix)])
    for candidate in candidates:
        # Most recipes put licences in info/licenses; the GCC runtime ships
        # them as package files under share/licenses instead.
        for licenses in (candidate / "info" / "licenses", candidate / "share" / "licenses"):
            if licenses.is_dir() and any(licenses.iterdir()):
                return licenses
    return None


def pip_packages(prefix: Path) -> list[dict]:
    packages = []
    for dist_info in sorted(prefix.glob("lib/python3*/site-packages/*.dist-info")):
        installer = dist_info / "INSTALLER"
        if not installer.is_file() or installer.read_text().strip() != "pip":
            continue
        metadata_file = dist_info / "METADATA"
        if not metadata_file.is_file():
            continue
        meta = Parser().parsestr(metadata_file.read_text(errors="replace"), headersonly=True)
        licence = meta.get("License-Expression") or (meta.get("License") or "").strip().split("\n")[0]
        if not licence:
            classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or []
                           if c.startswith("License ::")]
            licence = "; ".join(classifiers)
        packages.append({"name": meta.get("Name", dist_info.name), "version": meta.get("Version", ""),
                         "license": licence, "dist_info": str(dist_info)})
    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conda-root", required=True, type=Path, help="Conda installation root")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    root = args.conda_root.resolve()
    licenses_out = args.out / "licenses" / "conda"
    manifests_out = args.out / "manifests"
    licenses_out.mkdir(parents=True, exist_ok=True)
    manifests_out.mkdir(parents=True, exist_ok=True)

    missing = set()
    total = 0
    for env_name, prefix in conda_envs(root):
        records = conda_records(prefix)
        with open(manifests_out / f"conda-{env_name}.tsv", "w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["name", "version", "build", "license", "channel", "url", "sha256"])
            for record in records:
                writer.writerow([record.get("name"), record.get("version"), record.get("build"),
                                 record.get("license", ""), record.get("channel", ""),
                                 record.get("url", ""), record.get("sha256", "")])
                total += 1
                target = licenses_out / f"{record.get('name')}-{record.get('version')}"
                if target.exists():
                    continue
                source = license_dir(root, record)
                if source is None:
                    missing.add(f"{record.get('name')}-{record.get('version')}")
                    continue
                shutil.copytree(source, target)

        pips = pip_packages(prefix)
        if pips:
            with open(manifests_out / f"pip-{env_name}.tsv", "w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["name", "version", "license", "dist_info"])
                for package in pips:
                    writer.writerow([package["name"], package["version"], package["license"],
                                     package["dist_info"]])

    (manifests_out / "conda-missing-license-files.txt").write_text(
        "".join(f"{name}\n" for name in sorted(missing))
    )
    print(f"{total} conda packages recorded; licence files copied for "
          f"{len(list(licenses_out.iterdir()))}, missing for {len(missing)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
