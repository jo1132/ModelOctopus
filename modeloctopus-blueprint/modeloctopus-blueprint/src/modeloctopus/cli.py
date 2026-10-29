import argparse
import json
import sys

from pydantic import ValidationError
import yaml

from modeloctopus.application.doctor import doctor
from modeloctopus.application.planner import plan
from modeloctopus.core.schema import ModelOctopusError


def main() -> int:
    parser = argparse.ArgumentParser(prog="ef", description="ModelOctopus M0 planning tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("recipe")
    commands.add_parser("doctor").add_argument("--profile", default="core-ci")
    args = parser.parse_args()
    try:
        result = plan(args.recipe) if args.command == "plan" else doctor(args.profile)
        print(json.dumps(result, indent=2))
        return 1 if result.get("validation") == "failed" else 0
    except ModelOctopusError as exc:
        print(exc.report.model_dump_json(), file=sys.stderr)
    except (ValidationError, OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"code": "INVALID_REQUEST", "message": str(exc)}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
