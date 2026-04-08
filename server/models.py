"""
Pydantic models for the Invoice Extraction Environment.

Defines the Action and Observation types used for communication
between the agent and the environment.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InvoiceAction(BaseModel):
    """Action sent by the agent to the environment.

    Commands:
        - 'view_document': View the current document text
        - 'view_fields': View the required fields to extract
        - 'extract': Submit extracted fields (payload = JSON string)
        - 'get_feedback': Get feedback on the last extraction attempt
    """

    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        ...,
        description="Command to execute: 'view_document', 'view_fields', 'extract', or 'get_feedback'",
    )
    payload: str = Field(
        default="",
        description="JSON string payload (used with 'extract' command)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class InvoiceObservation(BaseModel):
    """Observation returned by the environment after each step.

    Contains the response text, task metadata, current score,
    and episode control signals (done, reward).
    """

    model_config = ConfigDict(extra="forbid")

    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: Optional[float] = Field(default=None, description="Reward signal [0.0-1.0]")
    text: str = Field(default="", description="Response text from the environment")
    task_name: str = Field(default="", description="Current task name")
    current_score: float = Field(default=0.0, description="Best score achieved so far")
    attempts_remaining: int = Field(default=0, description="Remaining extraction attempts")
    required_fields: List[str] = Field(default_factory=list, description="Fields to extract")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InvoiceState(BaseModel):
    """Internal environment state."""

    model_config = ConfigDict(extra="allow")

    episode_id: Optional[str] = Field(default=None, description="Current episode ID")
    step_count: int = Field(default=0, ge=0, description="Steps taken in current episode")
    task_name: str = Field(default="", description="Current task name")
    document_id: str = Field(default="", description="Current document ID")
    best_score: float = Field(default=0.0, description="Best extraction score so far")
    attempts_used: int = Field(default=0, description="Extraction attempts used")
    max_attempts: int = Field(default=3, description="Maximum extraction attempts")
