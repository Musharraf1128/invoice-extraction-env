"""
Pydantic models for the Enterprise Supply Chain & Tax Reconciliation Environment.

Defines the Action, Observation, and State types used for communication
between the agent and the environment. Designed for type-safe interaction
with an ERP-like tool suite.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action — what the agent sends to the environment
# ---------------------------------------------------------------------------

class ESCTRAction(BaseModel):
    """Action sent by the agent to the ESCTR environment.

    The agent operates as an autonomous financial controller using 4 tool verbs:
      - 'query_database': Search procurement, accounts payable, shipping, or warehouse databases
      - 'read_document': Retrieve a specific contract, SLA, PO, or invoice by document_id
      - 'communicate_vendor': Send a negotiation message to the simulated vendor
      - 'submit_financial_decision': Submit the final ledger adjustment (terminal action)
    """

    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "query_database",
        "read_document",
        "communicate_vendor",
        "submit_financial_decision",
    ] = Field(
        ...,
        description=(
            "The tool verb to execute. One of: 'query_database', 'read_document', "
            "'communicate_vendor', or 'submit_financial_decision'."
        ),
    )
    query_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured query for database lookups. Example: "
            '{"table": "shipping_logs", "tracking_id": "TRK-9921"}'
        ),
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Unique alphanumeric identifier of the document to read (e.g. 'PO-2024-0055').",
    )
    message_content: Optional[str] = Field(
        default=None,
        description="Natural language message for vendor negotiation (used with 'communicate_vendor').",
    )
    adjustment_amount: Optional[float] = Field(
        default=None,
        description=(
            "The precise monetary adjustment to submit (used with 'submit_financial_decision'). "
            "Must be the exact floating-point value calculated from contract terms."
        ),
    )
    adjustment_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of the adjustment rationale (used with 'submit_financial_decision').",
    )


# ---------------------------------------------------------------------------
# Observation — what the environment returns after each step
# ---------------------------------------------------------------------------

class ESCTRObservation(BaseModel):
    """Observation returned by the ESCTR environment after each step.

    Provides structured telemetry to help the agent understand the
    outcome of its action and plan the next move.
    """

    model_config = ConfigDict(extra="forbid")

    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Reward signal for this step (0.0-1.0)")
    system_response: str = Field(
        default="",
        description="Output from the tool: database results, document text, vendor reply, or grader feedback.",
    )
    last_action_status: Literal["success", "error"] = Field(
        default="success",
        description="Whether the last action was valid and executed successfully.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Diagnostic error message if last_action_status is 'error'.",
    )
    current_step: int = Field(
        default=0,
        description="Current step number within the episode (0-indexed at reset).",
    )
    max_steps: int = Field(
        default=15,
        description="Maximum steps allowed for this task.",
    )
    accumulated_reward: float = Field(
        default=0.0,
        description="Total reward accumulated across all steps in this episode.",
    )
    task_name: str = Field(default="", description="Current task name.")
    available_tools: List[str] = Field(
        default_factory=list,
        description="List of tool verbs available in this task.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata (scores, milestones, etc.).",
    )


# ---------------------------------------------------------------------------
# State — internal environment state (exposed via GET /state)
# ---------------------------------------------------------------------------

class ESCTRState(BaseModel):
    """Internal environment state for the ESCTR environment."""

    model_config = ConfigDict(extra="allow")

    episode_id: Optional[str] = Field(default=None, description="Current episode ID")
    step_count: int = Field(default=0, ge=0, description="Steps taken in current episode")
    task_name: str = Field(default="", description="Current task name")
    seed: int = Field(default=0, description="Seed used for procedural generation")
    accumulated_reward: float = Field(default=0.0, description="Total reward accumulated")
    outcome_submitted: bool = Field(default=False, description="Whether final decision was submitted")
    milestones_hit: List[str] = Field(
        default_factory=list,
        description="Trajectory milestones achieved (e.g. 'retrieved_po', 'retrieved_sla').",
    )
    best_score: float = Field(default=0.0, description="Best score achieved")
