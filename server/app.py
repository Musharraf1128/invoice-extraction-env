"""
FastAPI application for the Invoice Extraction Environment.

Exposes the environment over HTTP and WebSocket endpoints
compatible with the OpenEnv client protocol.
"""

import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models import InvoiceAction, InvoiceObservation, InvoiceState
from .environment import InvoiceExtractionEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models (OpenEnv-compatible)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: str = "simple_invoice"

    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None

    class Config:
        extra = "allow"


class HealthResponse(BaseModel):
    status: str = "healthy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_response(obs: InvoiceObservation) -> dict:
    """Convert an InvoiceObservation to a step/reset response dict."""
    obs_dict = obs.model_dump()
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    return {
        "observation": obs_dict,
        "reward": reward if reward is not None else 0.0,
        "done": done,
    }


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_invoice_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Invoice Extraction Environment",
        description="OpenEnv environment for extracting structured data from invoices",
        version="0.1.0",
    )

    # Global environment instance for HTTP endpoints
    _env = InvoiceExtractionEnvironment()

    # === Health check ===
    @app.get("/health")
    def health():
        return HealthResponse()

    @app.get("/")
    def root():
        return {
            "name": "invoice_extraction_env",
            "version": "0.1.0",
            "status": "running",
            "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/ws"],
        }

    # === Reset ===
    @app.post("/reset")
    def reset(request: ResetRequest = ResetRequest()):
        kwargs = request.model_dump(exclude_unset=True)
        obs = _env.reset(**kwargs)
        return _obs_to_response(obs)

    # === Step ===
    @app.post("/step")
    def step(request: StepRequest):
        try:
            action = InvoiceAction(**request.action)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"detail": f"Invalid action: {str(e)}"},
            )
        obs = _env.step(action, timeout_s=request.timeout_s)
        return _obs_to_response(obs)

    # === State ===
    @app.get("/state")
    def get_state():
        return _env.state.model_dump()

    # === Schema ===
    @app.get("/schema")
    def get_schema():
        return {
            "action": InvoiceAction.model_json_schema(),
            "observation": InvoiceObservation.model_json_schema(),
            "state": InvoiceState.model_json_schema(),
        }

    # === Metadata ===
    @app.get("/metadata")
    def get_metadata():
        return {
            "name": "invoice_extraction_env",
            "description": (
                "An environment for extracting structured data from unstructured "
                "invoice and receipt documents. Features 5 difficulty tiers from "
                "clean invoices to adversarial documents with decoy fields, OCR "
                "corruption, and hidden calculations. Includes procedural document "
                "generation for infinite training configurations, RLVR-inspired "
                "composite reward architecture with trajectory milestones, and "
                "multi-tool agentic workflow for complex tasks."
            ),
            "version": "0.3.0",
            "features": [
                "procedural_document_generation",
                "rlvr_composite_rewards",
                "multi_tool_workflow",
                "weighted_field_scoring",
                "cross_field_verification",
            ],
            "tasks": [
                {"name": "simple_invoice", "difficulty": "easy", "attempts": 3},
                {"name": "messy_invoice", "difficulty": "medium", "attempts": 3},
                {"name": "multi_document", "difficulty": "hard", "attempts": 5,
                 "tools": ["query_related_documents", "verify_calculations", "check_discrepancies"]},
                {"name": "corrupted_scan", "difficulty": "very_hard", "attempts": 4},
                {"name": "adversarial_invoice", "difficulty": "expert", "attempts": 6,
                 "tools": ["query_related_documents", "verify_calculations", "check_discrepancies"]},
            ],
        }

    # === WebSocket (for persistent sessions) ===
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        ws_env = InvoiceExtractionEnvironment()
        logger.info("WebSocket session opened")

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Invalid JSON", "code": "INVALID_JSON"},
                    })
                    continue

                msg_type = msg.get("type", "")
                msg_data = msg.get("data", {})

                if msg_type == "reset":
                    obs = ws_env.reset(**msg_data)
                    await websocket.send_json({
                        "type": "observation",
                        "data": _obs_to_response(obs),
                    })

                elif msg_type == "step":
                    try:
                        action = InvoiceAction(**msg_data)
                        obs = ws_env.step(action)
                        await websocket.send_json({
                            "type": "observation",
                            "data": _obs_to_response(obs),
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": str(e), "code": "EXECUTION_ERROR"},
                        })

                elif msg_type == "state":
                    await websocket.send_json({
                        "type": "state",
                        "data": ws_env.state.model_dump(),
                    })

                elif msg_type == "close":
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {msg_type}",
                            "code": "UNKNOWN_TYPE",
                        },
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket session disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            ws_env.close()
            logger.info("WebSocket session closed")

    return app


# Create the app instance
app = create_invoice_app()


def main():
    """Entry point for `uv run server` / `[project.scripts]`."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
