"""Run with python -m nwm; every experiment command shares a JSON config."""

import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Video-based physics experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "train", "evaluate", "demo", "check-device"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/smoke.json")
        if name != "generate":
            command.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
        if name == "train":
            command.add_argument("--stage", default="all", choices=("all", "perception", "decoder", "learned", "hybrid"))
            command.add_argument("--seed", type=int, action="append", help="Train only listed dynamics seeds")
            command.add_argument("--resume", action="store_true", help="Resume each stage from its last completed epoch")
    args = parser.parse_args(argv)
    if args.command == "check-device":
        from .training import device_check
        print(json.dumps(device_check(args.device), indent=2))
        return
    config = load_config(args.config)
    if args.command == "generate":
        import torch
        from .data import generate_dataset
        torch.set_num_threads(config["training"].get("cpu_threads", 4))
        output = generate_dataset(config)
    else:
        from .data import validate_dataset
        validate_dataset(config)
        if args.command == "train":
            from .training import train
            output = train(config, args.device, args.stage, args.resume, args.seed)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            output = evaluate(config, args.device)
        else:
            from .demo import make_demo
            output = make_demo(config, args.device)
    print(f"{args.command} complete: {output}", flush=True)


if __name__ == "__main__":
    main()
