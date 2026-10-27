#!/usr/bin/env python3
"""Check Cargo's source package before building it. Never publish anything.

Run from any directory: python3 scripts/check-package.py
Cargo uses the local cache only. Fetch locked dependencies before this check.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {f"prompts/{size}k.txt" for size in (1, 2, 4, 8, 16, 32, 64, 128, 256)}


def validate_names(names: set[str], required: set[str]) -> None:
    unsafe = sorted(
        name
        for name in names
        if PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or "models" in PurePosixPath(name).parts
        or name.lower().endswith((".gguf", ".safetensors", ".pt", ".pth"))
    )
    if unsafe:
        raise RuntimeError(f"Model weights or unsafe package paths: {unsafe}")
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Missing package source/assets: {missing}")


def run(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE if capture else None
    )


def main() -> None:
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    package = manifest["package"]
    prefix = f"{package['name']}-{package['version']}"
    required = PROMPTS | {"Cargo.toml", "Cargo.lock", "README.md", "LICENSE"}
    required |= {
        path.relative_to(ROOT).as_posix()
        for directory in (ROOT / "src", ROOT / "tests")
        for path in directory.rglob("*")
        if path.is_file()
    }
    # Fail before archiving if the manifest would include a large model.
    listing = run(
        "cargo", "package", "--list", "--locked", "--offline", "--allow-dirty", capture=True
    )
    names = set(listing.stdout.splitlines())
    validate_names(names, required)

    # Keep source archives and verification build output outside the checkout.
    # tempfile honors TMPDIR. In CI this is set to RUNNER_TEMP.
    with tempfile.TemporaryDirectory(prefix="autarky-package-") as temporary:
        target = Path(temporary)
        run(
            "cargo", "package", "--locked", "--offline", "--allow-dirty",
            "--target-dir", str(target),
        )
        archive = target / "package" / f"{prefix}.crate"
        with tarfile.open(archive, "r:gz") as packaged:
            files = {}
            for member in packaged.getmembers():
                if not member.isfile():
                    if not member.isdir():
                        raise RuntimeError(f"Unsupported package entry: {member.name}")
                    continue
                path = PurePosixPath(member.name)
                if not path.parts or path.parts[0] != prefix:
                    raise RuntimeError(f"Unexpected archive root: {member.name}")
                name = PurePosixPath(*path.parts[1:]).as_posix()
                if name in files:
                    raise RuntimeError(f"Duplicate archive entry: {name}")
                files[name] = member
            validate_names(set(files), required)
            for name in required - {"Cargo.toml", "Cargo.lock"}:
                source = packaged.extractfile(files[name])
                if source is None or source.read() != (ROOT / name).read_bytes():
                    raise RuntimeError(f"Packaged content differs from source: {name}")
        print(
            f"Package verified: {len(files)} files, {len(PROMPTS)} embedded prompt assets, "
            "all CLI source/tests, no model weights; Cargo verified the extracted build."
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f"Package check failed: {error}", file=sys.stderr)
        sys.exit(1)
