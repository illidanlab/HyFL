from datetime import datetime
import functools
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Scalar,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

# Can't import loguru's root logger in global scope
# This causes problems with multiprocessing
# We import inside classes/factories instead
# https://github.com/ray-project/ray/issues/14717
# https://stackoverflow.com/a/73711616
from loguru._logger import Logger


def serialize(record):
    """Serialize supervisor log record."""
    subset = {
        "time": record["time"].strftime("%Y-%m-%dT%H:%M:%S:%f%z"),
        "timestamp": record["time"].timestamp(),
        "message": record["message"],
        "cid": record["extra"].get("cid"),
        "method": record["extra"].get("method"),
        "event": record["extra"].get("event"),
        "captured_class": record["extra"].get("captured_class"),
        "captured_path": record["extra"].get("captured_path"),
    }
    return json.dumps(subset)


def formatter(record):
    """Format supervisor log record."""
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


def create_supervisor_logger(logger: Logger, log_path: Path):
    """Create supervisor logger with bound extra and handler."""
    # Create new child logger from root logger
    supervisor_logger = logger.bind(supervisor=True)
    # Add supervisor handler to new logger
    handler_id = supervisor_logger.add(
        log_path,
        filter=lambda record: record["extra"].get("supervisor"),
        format=formatter,
        enqueue=True,
    )
    return supervisor_logger, handler_id


class FederatedSupervisor:
    """Class that does client and filesystem path bookkeeping for the simulation."""

    base_storage_dir = Path("/code_execution/package")

    def __init__(self, partition_config_path: Union[str, Path]) -> None:
        # Set up paths
        partition_config_path = Path(partition_config_path)
        with partition_config_path.open("r") as fp:
            self.partition_config = json.load(fp)
        self.scenario_data_dir = partition_config_path.parent
        scenario_name, stage = partition_config_path.parts[-3:-1]

        # Captured directory for captured client—server communications
        self.base_captured_dir = (
            self.base_storage_dir / "captured" / scenario_name / stage
        )
        self.base_captured_dir.mkdir(exist_ok=True, parents=True)

        # State directory for saving client and server state
        self.base_state_dir = self.base_storage_dir / "state" / scenario_name
        self.base_state_dir.mkdir(exist_ok=True, parents=True)

        # Predictions directory for test predictions
        if stage == "test":
            self.base_predictions_dir = (
                self.base_storage_dir / "predictions" / scenario_name
            )
            self.base_predictions_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.base_predictions_dir = None

        # Set up logger
        self.supervisor_log_path = (
            self.base_storage_dir / f"{scenario_name}-{stage}.log"
        )

    def get_client_ids(self):
        return list(self.partition_config.keys())

    def get_client_data_dir(self, cid: str):
        return self.scenario_data_dir / cid

    def get_client_captured_path(
        self, cid: str, method: str, dataclass: str, counter: int
    ):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return (
            self.base_captured_dir
            / f"{timestamp}-{cid}-{counter:02d}-{method}.{dataclass}.pb"
        )

    def get_client_state_dir(self, cid: str):
        client_state_dir = self.base_state_dir / cid
        client_state_dir.mkdir(exist_ok=True)
        return client_state_dir

    def get_server_state_dir(self):
        server_state_dir = self.base_state_dir / "server"
        server_state_dir.mkdir(exist_ok=True)
        return server_state_dir

    def get_data_paths(self, cid: str) -> Dict[str, Path]:
        return {
            k: self.get_client_data_dir(cid) / v
            for k, v in self.partition_config[cid].items()
        }

    def get_predictions_format_path(self, cid: str):
        if self.base_predictions_dir is None:
            raise Exception("Shouldn't get predictions format path if not test")
        if os.environ.get("PACKAGE_TRACK") == "fincrime" and cid != "swift":
            return None
        return self.get_client_data_dir(cid) / f"predictions_format.csv"

    def get_predictions_dest_path(self, cid: str):
        if self.base_predictions_dir is None:
            raise Exception("Shouldn't get predictions dest path if not test")
        if os.environ.get("PACKAGE_TRACK") == "fincrime" and cid != "swift":
            return None
        return self.base_predictions_dir / f"{cid}.csv"


def wrap_client_method(ins_proto_fn, res_proto_fn):
    """Decorator factory for wrapping client methods. Requires input and output
    protobuf conversion functions to be specified."""

    def decorator(method: Callable):
        """Decorator that wraps client methods. Performs supervisor logging and captures
        serialized inputs and outputs."""
        signature = inspect.signature(method)
        input_annotation = list(signature.parameters.values())[-1].annotation
        return_annotation = signature.return_annotation

        @functools.wraps(method)
        def wrapped_method(self, ins):
            supervisor_logger = self.supervisor_logger.bind(
                method=method.__name__
            ).patch(lambda record: record.update(function=method.__name__))
            # Capture ins data
            ins_path = self.supervisor.get_client_captured_path(
                cid=self.cid,
                method=method.__name__,
                dataclass=input_annotation.__name__,
                counter=self._get_counter_value(),
            )
            with ins_path.open("wb") as fp:
                fp.write(ins_proto_fn(ins).SerializeToString())
            supervisor_logger.info(
                f"Client {self.cid}: {method.__name__} start",
                event="start",
                captured_class=input_annotation.__name__,
                captured_path=str(ins_path),
            )

            # Execute
            res = getattr(self.solution_client, method.__name__)(ins)

            # Capture res data
            res_path = self.supervisor.get_client_captured_path(
                cid=self.cid,
                method=method.__name__,
                dataclass=return_annotation.__name__,
                counter=self._get_counter_value(),
            )
            supervisor_logger.info(
                f"Client {self.cid}: {method.__name__} end",
                event="end",
                captured_class=return_annotation.__name__,
                captured_path=str(res_path),
            )
            with res_path.open("wb") as fp:
                fp.write(res_proto_fn(res).SerializeToString())
            return res

        return wrapped_method

    return decorator


class FederatedWrapperClient(fl.client.Client):
    """Wrapper around user-submitted solution client that implements standardized
    logging and communications capture.
    """

    def __init__(
        self,
        cid: str,
        solution_client: fl.client.ClientLike,
        supervisor: FederatedSupervisor,
        root_logger: Logger,
    ) -> None:
        self.cid = cid
        self.solution_client = solution_client
        self.supervisor = supervisor

        self.supervisor_logger, self.log_handler_id = create_supervisor_logger(
            logger=root_logger, log_path=supervisor.supervisor_log_path
        )
        self.supervisor_logger = self.supervisor_logger.bind(cid=cid)
        self.method_call_counter = 0
        self.supervisor_logger.info(
            f"Initializing Client {self.cid}.", method="__init__"
        )
        super().__init__()

    def __del__(self):
        self.supervisor_logger.info(f"Finalizing Client {self.cid}.", method="__del__")
        self.supervisor_logger.complete()
        self.supervisor_logger.remove(self.log_handler_id)

    def _get_counter_value(self):
        value = self.method_call_counter
        self.method_call_counter += 1
        return value

    @wrap_client_method(
        ins_proto_fn=fl.common.serde.get_properties_ins_to_proto,
        res_proto_fn=fl.common.serde.get_properties_res_to_proto,
    )
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        pass

    @wrap_client_method(
        ins_proto_fn=fl.common.serde.get_parameters_ins_to_proto,
        res_proto_fn=fl.common.serde.get_parameters_res_to_proto,
    )
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        pass

    @wrap_client_method(
        ins_proto_fn=fl.common.serde.fit_ins_to_proto,
        res_proto_fn=fl.common.serde.fit_res_to_proto,
    )
    def fit(self, ints: FitIns) -> FitRes:
        pass

    @wrap_client_method(
        ins_proto_fn=fl.common.serde.evaluate_ins_to_proto,
        res_proto_fn=fl.common.serde.evaluate_res_to_proto,
    )
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        pass


def wrap_train_client_factory(
    solution_client_factory: Callable[..., fl.client.Client],
    supervisor: FederatedSupervisor,
):
    """Wraps submitted train_client_factory from solution to control input args and
    to wrap instantiated client class."""

    def wrapped_client_factory(cid):
        from loguru import logger

        logger.info(f"Executing train_client_factory for Client {cid}...")
        solution_client = solution_client_factory(
            cid=cid,
            **supervisor.get_data_paths(cid),
            client_dir=supervisor.get_client_state_dir(cid),
        )
        return FederatedWrapperClient(
            cid=cid,
            solution_client=fl.client.to_client(solution_client),
            supervisor=supervisor,
            root_logger=logger,
        )

    return wrapped_client_factory


def wrap_test_client_factory(
    solution_client_factory: Callable[..., fl.client.Client],
    supervisor: FederatedSupervisor,
):
    """Wraps submitted test_client_factory from solution to control input args and
    to wrap instantiated client class."""

    def wrapped_client_factory(cid):
        from loguru import logger

        logger.info(f"Executing test_client_factory for Client {cid}...")
        solution_client = solution_client_factory(
            cid=cid,
            **supervisor.get_data_paths(cid),
            client_dir=supervisor.get_client_state_dir(cid),
            preds_format_path=supervisor.get_predictions_format_path(cid),
            preds_dest_path=supervisor.get_predictions_dest_path(cid),
        )
        return FederatedWrapperClient(
            cid=cid,
            solution_client=fl.client.to_client(solution_client),
            supervisor=supervisor,
            root_logger=logger,
        )

    return wrapped_client_factory


def wrap_strategy_method(method: Callable):
    """Decorator that wraps strategy methods. Performs supervisor logging and captures
    serialized inputs and outputs."""

    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        supervisor_logger = self.supervisor_logger.bind(method=method.__name__).patch(
            lambda record: record.update(function=method.__name__)
        )
        supervisor_logger.info(f"Strategy: {method.__name__} start", event="start")
        out = getattr(self.solution_strategy, method.__name__)(*args, **kwargs)
        supervisor_logger.info(f"Strategy: {method.__name__} end", event="end")
        return out

    return wrapped_method


class FederatedWrapperStrategy(fl.server.strategy.Strategy):
    """Wrapper around user-submitted solution strategy that implements standardized
    logging and communications capture.
    """

    def __init__(
        self,
        solution_strategy: fl.server.strategy.Strategy,
        supervisor: FederatedSupervisor,
    ) -> None:
        self.solution_strategy = solution_strategy
        self.supervisor = supervisor

        from loguru import logger

        self.supervisor_logger, self.log_handler_id = create_supervisor_logger(
            logger=logger, log_path=supervisor.supervisor_log_path
        )
        self.supervisor_logger = self.supervisor_logger.bind(cid="server")

        self.supervisor_logger.info(
            f"Initialized strategy {type(solution_strategy).__name__}.",
            method="__init__",
        )
        super().__init__()

    def __del__(self):
        self.supervisor_logger.info(f"Finalizing Strategy.", method="__del__")
        self.supervisor_logger.complete()

    @wrap_strategy_method
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        pass

    @wrap_strategy_method
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        pass

    @wrap_strategy_method
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        pass

    @wrap_strategy_method
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        pass

    @wrap_strategy_method
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        pass

    @wrap_strategy_method
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        pass


class CentralizedSupervisor:
    """Class that does path bookkeeping for centralized evaluation."""

    base_data_dir = Path("/code_execution/data/centralized")
    base_storage_dir = Path("/code_execution/package")

    def __init__(self, stage: str, root_logger: Logger) -> None:
        # Set up paths
        self.model_state_dir = self.base_storage_dir / "state" / "centralized"
        self.model_state_dir.mkdir(exist_ok=True, parents=True)
        self.stage_data_dir = self.base_data_dir / stage
        # Load data file paths from config file
        data_config_path = self.stage_data_dir / "data.json"
        with data_config_path.open("r") as fp:
            self.data_config = json.load(fp)

        if stage == "test":
            self.base_predictions_dir = (
                self.base_storage_dir / "predictions" / "centralized"
            )
            self.base_predictions_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.base_predictions_dir = None

        # Set up logger
        self.supervisor_log_path = self.base_storage_dir / f"centralized-{stage}.log"
        self.supervisor_logger, self.log_handler_id = create_supervisor_logger(
            logger=root_logger, log_path=self.supervisor_log_path
        )

    def get_model_state_dir(self):
        return self.model_state_dir

    def get_data_paths(self) -> Dict[str, Path]:
        return {k: self.stage_data_dir / v for k, v in self.data_config.items()}

    def get_predictions_format_path(self):
        return self.stage_data_dir / "predictions_format.csv"

    def get_predictions_dest_path(self):
        if self.base_predictions_dir is None:
            raise Exception("Shouldn't get predictions dest path if not test")
        return self.base_predictions_dir / "predictions.csv"
