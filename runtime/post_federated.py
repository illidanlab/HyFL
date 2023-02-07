import json
from pathlib import Path
import tarfile

from loguru import logger
import pandas as pd

from supervisor import FederatedSupervisor


OUTPUT_DIR = Path("/code_execution/package/scoring_payload/")
OUTPUT_TAR = Path("/code_execution/package/scoring_payload.tar.gz")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize metrics dict
    metrics = {}

    # Load memory data
    mem_df = pd.read_csv("package/memory_metrics.csv.gz", delimiter=";")

    with Path("/code_execution/data/scenarios.txt").open("r") as fp:
        scenarios = [line.strip() for line in fp if line.strip()]
    for scenario in scenarios:
        logger.info(f"Performing post-run for {scenario}...")
        train_supervisor = FederatedSupervisor(partition_config_path=f"/code_execution/data/{scenario}/train/partitions.json")
        test_supervisor = FederatedSupervisor(partition_config_path=f"/code_execution/data/{scenario}/test/partitions.json")

        # Collate predictions
        logger.info(f"Collating predictions for {scenario}...")
        predictions = []
        for cid in test_supervisor.get_client_ids():
            preds_path = test_supervisor.get_predictions_dest_path(cid)
            if preds_path is not None:
                predictions.append(pd.read_csv(preds_path))
        predictions_df = pd.concat(predictions)
        predictions_df.to_csv(OUTPUT_DIR / f"{scenario}_predictions.csv", index=False)

        # Calculate compute metrics
        logger.info(f"Retrieving runtime compute metrics for {scenario}...")
        logs_file = train_supervisor.supervisor_log_path
        logs_df = pd.read_json(logs_file, lines=True)
        timestamps = pd.to_datetime(logs_df["timestamp"])
        start, end = timestamps.min().tz_localize("utc"), timestamps.max().tz_localize("utc")
        duration = (end-start).total_seconds()
        peak_mem = mem_df[pd.to_datetime(mem_df["timestamp"]).between(start, end)]["kbcommit"].max()
        metrics[f"total_training_time_{scenario}"] = float(duration)
        metrics[f"peak_training_memory_kb_{scenario}"] = float(peak_mem)

        # Calculate network overheard metrics
        logger.info(f"Aggregating network overheard metrics for {scenario}...")
        num_files = 0
        total_disk = 0.0
        for captured_file in train_supervisor.base_captured_dir.glob("*.pb"):
            num_files += 1
            total_disk += captured_file.stat().st_size / 1024.0
        metrics[f"network_file_volume_{scenario}"] = num_files
        metrics[f"network_disk_volume_{scenario}"] = total_disk

    logger.info(f"Metrics summary:\n{json.dumps(metrics, indent=2)}")
    with Path("package/metrics.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)

    # Create predictions archive
    logger.info("Creating predictions tar file...")
    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        for scenario in scenarios:
            preds_file = OUTPUT_DIR / f"{scenario}_predictions.csv"
            tar.add(preds_file, arcname=preds_file.name)

    logger.info("Post-run complete.")
