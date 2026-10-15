"""Pipeline entrypoint with an asset-free configuration doctor."""
import argparse
import json
import sys
from .configuration import doctor, load_profile
from .pipeline import STAGES, improve, run_stage

def main(argv=None):
    parser = argparse.ArgumentParser(prog="evo-wam")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor")
    check.add_argument("--config", required=True)
    for name in (*STAGES, "idm-train", "idm-calibrate", "improve"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--round", type=int)
        command.add_argument("--dry-run", action="store_true")
        if name == "improve":
            command.add_argument("--rounds", type=int, default=4)
            command.add_argument("--from", dest="start", choices=STAGES, default="generate")
            command.add_argument("--to", dest="end", choices=STAGES, default="evaluate")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(args.config), indent=2))
            return 0
        config = load_profile(args.config)
        results = improve(config, args.round, args.rounds, args.start, args.end, args.dry_run) if args.command == "improve" else [run_stage(config, args.command, args.round or 1, args.dry_run)]
        for result in results: print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
