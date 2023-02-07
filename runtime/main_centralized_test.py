from loguru import logger

from supervisor import CentralizedSupervisor

from src.solution_centralized import predict


if __name__ == "__main__":
    logger.info(f"Starting {__file__}...")

    supervisor = CentralizedSupervisor("test", root_logger=logger)
    supervisor_logger = supervisor.supervisor_logger

    supervisor_logger.info(
        "Running provided predict function...",
        cid="centralized",
        method="predict",
        event="start",
    )
    predict(
        **supervisor.get_data_paths(),
        model_dir=supervisor.get_model_state_dir(),
        preds_format_path=supervisor.get_predictions_format_path(),
        preds_dest_path=supervisor.get_predictions_dest_path(),
    )
    supervisor_logger.info(
        "...done running predict",
        cid="centralized",
        method="predict",
        event="end",
    )

    # Post-validation
    logger.info("Validating that required predictions file exists...")
    assert supervisor.get_predictions_dest_path().exists()
