"""Prepare the existing published search results for the AI scripts; no cluster access."""
import argparse
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def prepare(results, destination):
    sources = {}
    for summary in sorted(results.glob("add*/summary.json")):
        metadata = json.loads(summary.read_text())
        cells = [5, 6] if summary.parent.name == "add2" else metadata["cells"]
        prefix = "adders_c" if len(cells) == 2 else "add3_c"
        name = prefix + "_".join(map(str, cells)) + ".txt"
        source = summary.parent / "adders_L5.txt"
        if source.exists():
            sources[name] = source
    catalog = results / "catalog/catalog.txt"
    if catalog.exists():
        sources["catalog.txt"] = catalog
    if not sources:
        raise FileNotFoundError(f"No published search results found under {results}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = destination / name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Refusing to replace different existing data: {target}")
        else:
            shutil.copyfile(source, target)
    return sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parent / "results")
    parser.add_argument("--destination", type=Path, default=HERE / "data")
    args = parser.parse_args()
    files = prepare(args.results, args.destination)
    print(f"Prepared {len(files)} datasets in {args.destination}")
