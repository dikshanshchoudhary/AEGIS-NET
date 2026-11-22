from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI

from api.routes.detections import router
from database.repository import DetectionRepository
from imb.detection.engine import DetectionEngine
from imb.utils.config import PROJECT_ROOT, load_config
from imb.utils.logging_config import configure_logging


def _configured_path(value: object, setting_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration setting {setting_name!r} must be a path.")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def create_app(
    *,
    engine_loader: Callable[[], DetectionEngine] = DetectionEngine.load,
    database_path: Path | None = None,
    evaluation_metrics_path: Path | None = None,
) -> FastAPI:
    config = load_config()
    app_config = config.get("app")
    path_config = config.get("paths")
    api_config = config.get("api")
    logging_config = config.get("logging")
    if not isinstance(app_config, dict) or not isinstance(path_config, dict):
        raise ValueError("Application config must include app and paths mappings.")
    if not isinstance(api_config, dict) or not isinstance(logging_config, dict):
        raise ValueError("Application config must include api and logging mappings.")

    resolved_database_path = database_path or _configured_path(
        path_config.get("database"),
        "paths.database",
    )
    resolved_metrics_path = evaluation_metrics_path or (
        PROJECT_ROOT / "artifacts" / "preprocessing" / "evaluation_metrics.json"
    )
    max_upload_bytes = api_config.get("max_upload_bytes")
    max_csv_rows = api_config.get("max_csv_rows")
    if (
        not isinstance(max_upload_bytes, int)
        or max_upload_bytes < 1
        or not isinstance(max_csv_rows, int)
        or max_csv_rows < 1
    ):
        raise ValueError("API upload byte and CSV row limits must be positive integers.")
    logging_level = logging_config.get("level")
    if not isinstance(logging_level, str):
        raise ValueError("Configuration setting 'logging.level' must be a string.")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging(logging_level)
        repository = DetectionRepository(resolved_database_path)
        repository.initialize()
        engine = engine_loader()
        with resolved_metrics_path.open("r", encoding="utf-8") as source:
            evaluation_metrics = json.load(source)
        if not isinstance(evaluation_metrics, dict):
            raise ValueError("Evaluation metrics artifact must contain a JSON object.")

        application.state.repository = repository
        application.state.engine = engine
        application.state.evaluation_metrics = evaluation_metrics
        application.state.max_upload_bytes = max_upload_bytes
        application.state.max_csv_rows = max_csv_rows
        yield

    application = FastAPI(
        title=str(app_config.get("name", "IMB")),
        version=str(app_config.get("version", "0.1.0")),
        description=(
            "Offline network-flow classification and anomaly analysis. "
            "This service does not block traffic."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
