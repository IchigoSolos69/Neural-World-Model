"""Create a small portable source bundle, optionally carrying an experiment."""

import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nwm.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument("--output", default="nwm_colab.zip")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    sources = [ROOT / name for name in ("nwm", "configs", "notebooks", "tools", "tests", "requirements")]
    sources += [ROOT / name for name in ("README.md", "pyproject.toml", ".gitignore", "run.bat")]
    if args.include_artifacts:
        sources.extend(Path(config[section]["root"]).resolve() for section in ("data", "outputs"))
    paths = set()
    for source in sources:
        if not source.exists():
            continue
        for path in source.rglob("*") if source.is_dir() else [source]:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".partial"):
                if not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f"Bundle inputs must be inside the project: {path}")
                paths.add(path)
    output = Path(args.output).resolve()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.resolve() != output:
                archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Created {output} ({output.stat().st_size / 2**20:.1f} MiB, {len(paths)} files)")


if __name__ == "__main__":
    main()
