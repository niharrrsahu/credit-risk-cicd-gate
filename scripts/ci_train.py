"""Standalone training runner for GitHub Actions CI."""
import json
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config_schema import load_config
from src.train import train_and_track

def main():
    print("Loading config/ci.yaml for candidate model training...")
    cfg = load_config("config/ci.yaml")
    res = train_and_track(cfg, run_name="github_actions_ci_run")
    with open("candidate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res["metrics"], f, indent=2)
    print("Candidate metrics successfully generated:", res["metrics"])

if __name__ == "__main__":
    main()
