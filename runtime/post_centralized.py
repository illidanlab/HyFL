import json
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd

from supervisor import CentralizedSupervisor


INPUT_FILE = Path("/code_execution/package/predictions/centralized/predictions.csv")
OUTPUT_FILE = Path("/code_execution/package/scoring_payload/predictions.csv")

if __name__ == "__main__":
    # Initialize metrics dict
    metrics = {}

    # Load memory data
    mem_df = pd.read_csv("package/memory_metrics.csv.gz", delimiter=";")

    logger.info(f"Performing post-run for centralized...")
    train_supervisor = CentralizedSupervisor("train", root_logger=logger)

    # Calculate compute metrics
    logger.info(f"Retrieving runtime compute metrics for centralized...")
    logs_file = train_supervisor.supervisor_log_path
    logs_df = pd.read_json(logs_file, lines=True)
    timestamps = pd.to_datetime(logs_df["timestamp"])
    start, end = timestamps.min().tz_localize("utc"), timestamps.max().tz_localize("utc")
    duration = (end-start).total_seconds()
    peak_mem = mem_df[pd.to_datetime(mem_df["timestamp"]).between(start, end)]["kbcommit"].max()
    metrics[f"total_training_time_centralized"] = float(duration)
    metrics[f"peak_training_memory_kb_centralized"] = float(peak_mem)

    logger.info(f"Metrics summary:\n{json.dumps(metrics, indent=2)}")
    with Path("package/metrics.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)

    # Copy predictions file
    logger.info("Collating test predictions...")
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(INPUT_FILE, OUTPUT_FILE)

    logger.info("Post-run complete.")
