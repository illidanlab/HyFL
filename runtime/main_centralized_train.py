from loguru import logger

from supervisor import CentralizedSupervisor

import src.solution_centralized as solution_centralized


if __name__ == "__main__":
    logger.info(f"Starting {__file__}...")

    # Pre-validation
    logger.info(
        "Validating that all required functions exist in solution_centralized..."
    )
    assert hasattr(solution_centralized, "fit") and callable(solution_centralized.fit)
    assert hasattr(solution_centralized, "predict") and callable(
        solution_centralized.predict
    )

    supervisor = CentralizedSupervisor("train", root_logger=logger)
    supervisor_logger = supervisor.supervisor_logger

    supervisor_logger.info(
        "Running provided fit function...",
        cid="centralized",
        method="predict",
        event="start",
    )
    solution_centralized.fit(
        **supervisor.get_data_paths(), model_dir=supervisor.get_model_state_dir()
    )
    supervisor_logger.info(
        "...done running fit",
        cid="centralized",
        method="fit",
        event="end",
    )
