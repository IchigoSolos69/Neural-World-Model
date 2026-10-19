"""Run with python -m nwm; every experiment command shares a JSON config."""

import argparse
import json

from .config import load_config


def _local_app_ready(url):
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
            health = json.loads(response.read(65536))
        return health.get("status") == "ok" and "worker" in health
    except (OSError, ValueError):
        return False


def _open_app_when_ready(url):
    import time
    import webbrowser
    for _ in range(60):
        if _local_app_ready(url):
            webbrowser.open(url)
            return
        time.sleep(.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Video-based physics experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("report")
    command.add_argument("--evaluation", required=True)
    command = commands.add_parser("release-research")
    command.add_argument("--config", default="configs/full.json")
    command.add_argument("--output", required=True)
    command.add_argument("--bundle", required=True)
    command.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    command.add_argument("--wait", action="store_true")
    for name in ("export-serving", "run-experiment", "experiment-status"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/full.json")
        if name == "export-serving":
            command.add_argument("--output", required=True)
            command.add_argument("--evaluation", help="Bind this bundle to a completed enhanced evaluation")
        if name == "run-experiment":
            command.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
            command.add_argument("--resume", action="store_true")
    command = commands.add_parser("serve")
    command.add_argument("--config", default="configs/app.json")
    command.add_argument("--port", type=int)
    command.add_argument("--open-browser", action="store_true")
    command = commands.add_parser("prepare-real-data")
    command.add_argument("--config", default="configs/app.json")
    command.add_argument("--download", action="store_true", help="Download the declared public evaluation mini set")
    command = commands.add_parser("benchmark-video")
    command.add_argument("--config", default="configs/app.json")
    command.add_argument("--manifest", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--split", choices=("development", "test"), default="test")
    for name in ("video-check", "video-download", "video-feasibility"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/app.json")
        command.add_argument("--output", default="runs/video-preflight.json")
        if name == "video-feasibility":
            command.add_argument("--prefix", action="append", required=True)
    for name in ("generate", "train", "evaluate", "demo", "check-device"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/smoke.json")
        if name != "generate":
            command.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
        if name == "train":
            command.add_argument("--stage", default="all", choices=("all", "perception", "decoder", "learned", "hybrid"))
            command.add_argument("--seed", type=int, action="append", help="Train only listed dynamics seeds")
            command.add_argument("--resume", action="store_true", help="Resume each stage from its last completed epoch")
        if name == "evaluate":
            command.add_argument("--output", help="Fresh directory for enhanced episode-level research evidence")
    args = parser.parse_args(argv)
    if args.command == "report":
        from .research_report import build_report
        print(build_report(args.evaluation))
        return
    if args.command == "release-research":
        from .research_release import release_research
        print(release_research(args.config, args.output, args.bundle, args.device, args.wait))
        return
    if args.command in ("prepare-real-data", "benchmark-video"):
        from pathlib import Path
        from .app_storage import app_config
        from .video_benchmark import prepare_mini, evaluate_mini
        settings = app_config(args.config)
        if args.command == "prepare-real-data":
            print(prepare_mini(settings, download=args.download))
        else:
            from .experiment import gpu_lock, experiment_status, ResourceBusyError
            current = experiment_status("configs/full.json")
            if current.get("state") == "running" or current.get("child_alive"):
                raise ResourceBusyError("Full training is using the GPU; benchmark after it finishes")
            with gpu_lock():
                result = evaluate_mini(args.manifest, Path(settings["root"]) / "models/ltx-2b-0.9.8",
                                       args.output, args.split)
            print(json.dumps(result, indent=2))
        return
    if args.command == "serve":
        import uvicorn
        from .app import create_app
        from .app_storage import app_config
        settings = app_config(args.config)
        port = args.port or settings.get("port", 8000)
        if not 1024 <= port <= 65535:
            parser.error("Choose a port between 1024 and 65535")
        if args.open_browser:
            import threading
            import webbrowser
            url = f"http://127.0.0.1:{port}"
            if _local_app_ready(url):
                webbrowser.open(url)
                return
            threading.Thread(target=_open_app_when_ready, args=(url,), daemon=True).start()
        uvicorn.run(create_app(settings), host="127.0.0.1", port=port, workers=1)
        return
    if args.command.startswith("video-"):
        from pathlib import Path
        from .app_storage import app_config
        from .video_model import asset_preflight, download_assets, run_feasibility
        settings = app_config(args.config)
        if args.command == "video-download":
            print(download_assets(settings["root"]))
        else:
            if args.command == "video-check":
                result = asset_preflight(settings["root"], quota_bytes=int(settings["limit_gib"] * 1024**3),
                                         reserve_bytes=int(settings["reserve_gib"] * 1024**3))
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            else:
                result = run_feasibility(settings["root"], args.prefix, args.output)
            print(json.dumps(result, indent=2))
        return
    if args.command == "experiment-status":
        from .experiment import experiment_status
        print(json.dumps(experiment_status(args.config), indent=2))
        return
    if args.command == "run-experiment":
        from .experiment import run_experiment
        print(run_experiment(args.config, args.device, args.resume))
        return
    if args.command == "export-serving":
        from .serving import export_serving
        print(export_serving(load_config(args.config), args.output, evaluation=args.evaluation))
        return
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
        if args.command == "train":
            from .training import train
            output = train(config, args.device, args.stage, args.resume, args.seed)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            output = evaluate(config, args.device, output=args.output)
        else:
            from .demo import make_demo
            output = make_demo(config, args.device)
    print(f"{args.command} complete: {output}", flush=True)


if __name__ == "__main__":
    main()
