"""Record the currently inspected Python environment, not a historical lock."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-on", required=True)
    args = parser.parse_args()
    packages = dict(sorted((d.metadata["Name"], d.version)
                           for d in importlib.metadata.distributions() if d.metadata["Name"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = [f"# Environment inspected on {args.verified_on}; Python {platform.python_version()}.",
              "# A current runtime snapshot, not evidence of a lock recorded at original training time."]
    args.output.write_text("\n".join(header + [f"{k}=={v}" for k,v in packages.items()])+"\n",encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps({
        "verified_on":args.verified_on,"python":platform.python_version(),
        "system":platform.system(),"packages":packages},indent=2)+"\n",encoding="utf-8")
    print(f"Recorded {len(packages)} package versions in {args.output}")


if __name__ == "__main__":
    main()
