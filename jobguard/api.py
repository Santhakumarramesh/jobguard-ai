"""FastAPI inference service for JobGuard AI."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import DEFAULT_THRESHOLD, MODEL_DIR
from .detector import load_detector
from .text import compose_job_text


class JobInput(BaseModel):
    text: str | None = None
    title: str | None = None
    company: str | None = None
    company_profile: str | None = None
    description: str | None = None
    requirements: str | None = None
    benefits: str | None = None
    has_company_logo: bool = False


class BatchJobInput(BaseModel):
    items: list[JobInput] = Field(default_factory=list)


def _compose_payload_text(payload: JobInput) -> str:
    if payload.text and payload.text.strip():
        return payload.text.strip()

    company_text = payload.company_profile or payload.company
    return compose_job_text(
        payload.title,
        company_text,
        payload.description,
        payload.requirements,
        payload.benefits,
    )


def _payload_to_frame(payload: JobInput) -> pd.DataFrame:
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_none=True)
    else:  # pragma: no cover - pydantic v1 fallback
        data = payload.dict(exclude_none=True)

    raw_text = data.pop("text", None)
    if data.get("company_profile") is None and data.get("company"):
        data["company_profile"] = data["company"]
    if raw_text and not any(data.get(field) for field in ("title", "company", "company_profile", "description", "requirements", "benefits")):
        data["description"] = raw_text
    elif raw_text and not data.get("description"):
        data["description"] = raw_text

    data["has_company_logo"] = int(bool(data.get("has_company_logo", False)))
    return pd.DataFrame([data])


def _predict_payload(detector, payload: JobInput) -> dict[str, Any]:
    structured_fields_present = any(
        getattr(payload, field) for field in ("title", "company", "company_profile", "description", "requirements", "benefits")
    )
    if structured_fields_present or payload.has_company_logo:
        frame = _payload_to_frame(payload)
        return detector.predict_dataframe(frame).iloc[0].to_dict()

    text = _compose_payload_text(payload)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Provide text or job fields to analyze.")
    return detector.predict(text)


def create_app(detector=None) -> FastAPI:
    """Create the ASGI app with a ready-to-use detector instance."""

    detector = detector or load_detector(MODEL_DIR)
    app = FastAPI(title="JobGuard AI Inference API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.detector = detector

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "JobGuard AI Inference API",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "detector_source": getattr(app.state.detector, "source", "unknown"),
            "threshold": getattr(app.state.detector, "threshold", DEFAULT_THRESHOLD),
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        return {
            "loaded": True,
            "detector_source": getattr(app.state.detector, "source", "unknown"),
            "threshold": getattr(app.state.detector, "threshold", DEFAULT_THRESHOLD),
            "artifact_dir": str(MODEL_DIR),
        }

    @app.post("/predict")
    def predict(payload: JobInput) -> dict[str, Any]:
        result = _predict_payload(app.state.detector, payload)
        result["model_source"] = getattr(app.state.detector, "source", "unknown")
        result["input_length"] = len(_compose_payload_text(payload))
        result["has_company_logo"] = bool(payload.has_company_logo)
        return result

    @app.post("/predict/batch")
    def predict_batch(payload: BatchJobInput) -> dict[str, Any]:
        if not payload.items:
            raise HTTPException(status_code=400, detail="Provide at least one item.")

        outputs = []
        for item in payload.items:
            output = _predict_payload(app.state.detector, item)
            output["model_source"] = getattr(app.state.detector, "source", "unknown")
            output["input_length"] = len(_compose_payload_text(item))
            outputs.append(output)

        return {"count": len(outputs), "items": outputs}

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run("jobguard.api:app", host="0.0.0.0", port=8000, reload=False)
