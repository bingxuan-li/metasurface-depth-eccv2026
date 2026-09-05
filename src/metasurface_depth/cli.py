"""One entry point; heavyweight optional dependencies load only when requested."""

import argparse
import importlib
import sys

COMMANDS = {
    "simulate": "simulate",
    "infer": "infer",
    "train": "train",
    "evaluate": "evaluate",
    "demo-data": "demo",
    "pipeline": "pipeline",
    "export": "export",
}


def main():
    parser = argparse.ArgumentParser(
        description="Metasurface depth: simulate, train, evaluate, infer"
    )
    parser.add_argument("command", choices=COMMANDS)
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"metasurface {args.command}", *sys.argv[2:]]
    importlib.import_module(f"metasurface_depth.{COMMANDS[args.command]}").main()
