import os
import sys
from pathlib import Path

import flwr as fl

from loguru import logger
from supervisor import (
    create_supervisor_logger,
    FederatedSupervisor,
    FederatedWrapperStrategy,
    wrap_train_client_factory,
)

import src.solution_federated as solution_federated


if __name__ == "__main__":
    logger.info(f"Starting {__file__}...")

    # Pre-validation
    logger.info("Validating that all required functions exist in solution_federated...")
    assert hasattr(solution_federated, "train_client_factory") and callable(
        solution_federated.train_client_factory
    )
    assert hasattr(solution_federated, "train_strategy_factory") and callable(
        solution_federated.train_strategy_factory
    )
    assert hasattr(solution_federated, "test_client_factory") and callable(
        solution_federated.test_client_factory
    )
    assert hasattr(solution_federated, "test_strategy_factory") and callable(
        solution_federated.test_strategy_factory
    )

    supervisor = FederatedSupervisor(partition_config_path=Path(sys.argv[1]))

    # Run optional train_setup function
    if hasattr(solution_federated, "train_setup"):
        supervisor_logger, log_handler_id = create_supervisor_logger(
            logger=logger, log_path=supervisor.supervisor_log_path
        )
        supervisor_logger.info(
            "train_setup found. Running...",
            cid="setup",
            method="train_setup",
            event="start",
        )
        solution_federated.train_setup(
            server_dir=supervisor.get_server_state_dir(),
            client_dirs_dict={
                cid: supervisor.get_client_state_dir(cid)
                for cid in supervisor.get_client_ids()
            },
        )
        supervisor_logger.info(
            "train_setup done.",
            cid="setup",
            method="train_setup",
            event="end",
        )
        supervisor_logger.complete()
        supervisor_logger.remove(log_handler_id)

    wrapped_client_factory = wrap_train_client_factory(
        solution_federated.train_client_factory, supervisor
    )
    solution_strategy, num_rounds = solution_federated.train_strategy_factory(
        server_dir=supervisor.get_server_state_dir()
    )
    wrapped_strategy = FederatedWrapperStrategy(
        solution_strategy=solution_strategy, supervisor=supervisor
    )
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    # Set CPUs more than half of available to prevent multiple clients from running
    # concurrently. Only used for scheduling—will not limit actual CPU usage.
    client_resources = {
        "num_cpus": int(os.cpu_count() / 2) + 1,
    }
    if os.getenv("CPU_OR_GPU", "") == "gpu":
        client_resources["num_gpus"] = 1

    # start simulation
    fl.simulation.start_simulation(
        client_fn=wrapped_client_factory,
        clients_ids=supervisor.get_client_ids(),
        client_resources=client_resources,
        config=server_config,
        strategy=wrapped_strategy,
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
        },
    )
