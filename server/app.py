"""
FastAPI application for the ESCTR Environment.

Exposes the Enterprise Supply Chain & Tax Reconciliation environment
over HTTP and WebSocket endpoints compatible with the OpenEnv spec.
"""

import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models import ESCTRAction, ESCTRObservation, ESCTRState
from .environment import ESCTREnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: str = "procurement_reconciliation"

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

def _obs_to_response(obs: ESCTRObservation) -> dict:
    obs_dict = obs.model_dump()
    reward = obs_dict.pop("reward", 0.0)
    done = obs_dict.pop("done", False)
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="ESCTR Environment",
        description=(
            "Enterprise Supply Chain & Tax Reconciliation — an OpenEnv environment "
            "for training LLMs to investigate discrepancies, enforce SLA penalties, "
            "and navigate adversarial vendor disputes."
        ),
        version="1.0.0",
    )

    _env = ESCTREnvironment()

    @app.get("/health")
    def health():
        return HealthResponse()

    @app.get("/")
    def root():
        return {
            "name": "esctr_environment",
            "version": "1.0.0",
            "status": "running",
            "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/metadata", "/ws"],
        }

    @app.post("/reset")
    def reset(request: ResetRequest = ResetRequest()):
        kwargs = request.model_dump(exclude_unset=True)
        obs = _env.reset(**kwargs)
        return _obs_to_response(obs)

    @app.post("/step")
    def step(request: StepRequest):
        try:
            action = ESCTRAction(**request.action)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"detail": f"Invalid action: {str(e)}"},
            )
        obs = _env.step(action, timeout_s=request.timeout_s)
        return _obs_to_response(obs)

    @app.get("/state")
    def get_state():
        return _env.state.model_dump()

    @app.get("/schema")
    def get_schema():
        return {
            "action": ESCTRAction.model_json_schema(),
            "observation": ESCTRObservation.model_json_schema(),
            "state": ESCTRState.model_json_schema(),
        }

    @app.get("/metadata")
    def get_metadata():
        return {
            "name": "esctr_environment",
            "description": (
                "Enterprise Supply Chain & Tax Reconciliation: an environment where "
                "an LLM agent operates as an autonomous financial controller, investigating "
                "procurement discrepancies, enforcing SLA penalties from shipping delays, "
                "and navigating adversarial vendor disputes. Features procedural generation "
                "for infinite scenarios, RLVR composite rewards, and multi-tool agentic workflow."
            ),
            "version": "1.0.0",
            "themes": [
                "World Modeling — Professional Tasks",
                "Long-Horizon Planning & Instruction Following",
                "Multi-Agent Interactions (adversarial vendor)",
            ],
            "tasks": [
                {"name": "procurement_reconciliation", "difficulty": "easy", "max_steps": 10,
                 "description": "Identify overcharged line items between PO and Invoice"},
                {"name": "sla_enforcement", "difficulty": "medium", "max_steps": 15,
                 "description": "Calculate late delivery penalties from shipping logs and SLA contracts"},
                {"name": "adversarial_auditing", "difficulty": "hard", "max_steps": 20,
                 "description": "Navigate vendor disputes, verify warehouse logs, reject settlement offers"},
            ],
            "tools": [
                "query_database", "read_document", "communicate_vendor", "submit_financial_decision",
            ],
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        ws_env = ESCTREnvironment()
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
                    await websocket.send_json({"type": "observation", "data": _obs_to_response(obs)})

                elif msg_type == "step":
                    try:
                        action = ESCTRAction(**msg_data)
                        obs = ws_env.step(action)
                        await websocket.send_json({"type": "observation", "data": _obs_to_response(obs)})
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": str(e), "code": "EXECUTION_ERROR"},
                        })

                elif msg_type == "state":
                    await websocket.send_json({"type": "state", "data": ws_env.state.model_dump()})

                elif msg_type == "close":
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Unknown message type: {msg_type}", "code": "UNKNOWN_TYPE"},
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket session disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            ws_env.close()
            logger.info("WebSocket session closed")

    return app


app = create_app()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
