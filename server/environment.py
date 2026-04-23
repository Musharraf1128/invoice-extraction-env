"""
ESCTR Environment — Core Implementation.

Enterprise Supply Chain & Tax Reconciliation: a stateful environment
where an LLM agent operates as an autonomous financial controller,
using ERP tools to investigate discrepancies, enforce SLA penalties,
and navigate adversarial vendor disputes.

Reward Architecture:
    R_total = α·R_outcome + β·R_trajectory − penalties
"""

import json
from dataclasses import asdict
from typing import Any, Optional
from uuid import uuid4

from .models import ESCTRAction, ESCTRObservation, ESCTRState
from .procedural import (
    generate_scenario, Scenario, VALID_TASKS, MAX_STEPS,
    render_purchase_order, render_invoice, render_sla,
    render_shipping_log, render_warehouse_logs,
)
from .graders import grade_task1, grade_task2, grade_task3

# Reward constants
STEP_COST = 0.005
HALLUCINATION_PENALTY = 0.02

# Available tools per task
TASK_TOOLS = {
    "procurement_reconciliation": [
        "query_database", "read_document", "submit_financial_decision",
    ],
    "sla_enforcement": [
        "query_database", "read_document", "submit_financial_decision",
    ],
    "adversarial_auditing": [
        "query_database", "read_document", "communicate_vendor", "submit_financial_decision",
    ],
}

# Database tables per task
AVAILABLE_TABLES = {
    "procurement_reconciliation": ["purchase_orders", "invoices"],
    "sla_enforcement": ["purchase_orders", "invoices", "shipping_logs", "sla_contracts"],
    "adversarial_auditing": ["purchase_orders", "invoices", "shipping_logs", "sla_contracts", "warehouse_logs"],
}


class ESCTREnvironment:
    """Enterprise Supply Chain & Tax Reconciliation Environment."""

    def __init__(self):
        self._state = ESCTRState(episode_id=str(uuid4()))
        self._scenario: Optional[Scenario] = None
        self._initialized = False
        self._trajectory_reward = 0.0
        self._milestones: list = []
        self._vendor_negotiation_count = 0
        self._settlement_offered = False
        self._settlement_rejected = False
        self._cited_evidence = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "procurement_reconciliation",
        **kwargs: Any,
    ) -> ESCTRObservation:
        """Reset the environment with a new scenario."""
        if task_name not in VALID_TASKS:
            task_name = "procurement_reconciliation"

        actual_seed = seed if seed is not None else 0
        scenario = generate_scenario(task_name, actual_seed)
        max_steps = MAX_STEPS.get(task_name, 15)

        self._state = ESCTRState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            seed=actual_seed,
            accumulated_reward=0.0,
            outcome_submitted=False,
            milestones_hit=[],
        )
        self._scenario = scenario
        self._initialized = True
        self._trajectory_reward = 0.0
        self._milestones = []
        self._vendor_negotiation_count = 0
        self._settlement_offered = False
        self._settlement_rejected = False
        self._cited_evidence = False

        tools = TASK_TOOLS.get(task_name, [])
        tables = AVAILABLE_TABLES.get(task_name, [])

        # Build initial briefing
        briefing = self._build_briefing(task_name, scenario, tables)

        return ESCTRObservation(
            done=False,
            reward=0.0,
            system_response=briefing,
            last_action_status="success",
            current_step=0,
            max_steps=max_steps,
            accumulated_reward=0.0,
            task_name=task_name,
            available_tools=tools,
        )

    def _build_briefing(self, task_name: str, scenario: Scenario, tables: list) -> str:
        """Generate task-specific initial briefing."""
        vendor = scenario.vendor.name
        buyer = scenario.buyer.name
        inv_num = scenario.invoice.invoice_number
        po_num = scenario.purchase_order.po_number

        if task_name == "procurement_reconciliation":
            return (
                f"=== DISCREPANCY ALERT ===\n"
                f"A pricing discrepancy has been detected between Purchase Order {po_num} "
                f"and Vendor Invoice {inv_num} from {vendor}.\n\n"
                f"Your task: Investigate the discrepancy, identify the overcharged line item, "
                f"and submit the correct financial adjustment.\n\n"
                f"Available databases: {', '.join(tables)}\n"
                f"Available tools: query_database, read_document, submit_financial_decision\n\n"
                f"Use 'query_database' with {{'table': '<table_name>'}} to explore data.\n"
                f"Use 'read_document' with document_id (e.g. '{po_num}' or '{inv_num}') to read full documents.\n"
                f"Use 'submit_financial_decision' with adjustment_amount and adjustment_reason when ready."
            )
        elif task_name == "sla_enforcement":
            return (
                f"=== PAYMENT DEMAND REVIEW ===\n"
                f"Vendor {vendor} has submitted Invoice {inv_num} (ref: {po_num}) "
                f"demanding full payment without penalties.\n\n"
                f"Intelligence suggests the shipment may have been delivered late. "
                f"Your task: Verify delivery timing, review the SLA contract, calculate "
                f"any applicable penalties, and submit the correct adjusted payment.\n\n"
                f"Available databases: {', '.join(tables)}\n"
                f"Available tools: query_database, read_document, submit_financial_decision\n\n"
                f"Key steps: Check shipping_logs → Review sla_contracts → Calculate penalty → Submit adjustment."
            )
        elif task_name == "adversarial_auditing":
            return (
                f"=== VENDOR DISPUTE ALERT ===\n"
                f"Vendor {vendor} has submitted Invoice {inv_num} (ref: {po_num}) "
                f"demanding full payment. Shipping records indicate a late delivery.\n\n"
                f"⚠ The vendor DISPUTES the late delivery claim. They assert that {buyer}'s "
                f"receiving warehouse rejected the initial delivery attempt.\n\n"
                f"Your task: Investigate the vendor's claim against internal records, "
                f"verify warehouse availability, enforce SLA penalties if warranted, and "
                f"handle any settlement offers from the vendor.\n\n"
                f"Available databases: {', '.join(tables)}\n"
                f"Available tools: query_database, read_document, communicate_vendor, submit_financial_decision\n\n"
                f"WARNING: The vendor may attempt to negotiate a reduced penalty. "
                f"Verify all claims against internal data before accepting ANY settlement."
            )
        return "Environment ready."

    def step(
        self,
        action: ESCTRAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ESCTRObservation:
        """Execute one step in the environment."""
        if not self._initialized:
            return self._error_obs("Environment not initialized. Call reset() first.", terminal=True)

        if self._state.outcome_submitted:
            return self._error_obs("Episode already complete. Call reset() for a new episode.", terminal=True)

        self._state.step_count += 1
        max_steps = MAX_STEPS.get(self._state.task_name, 15)

        # Step cost
        self._trajectory_reward -= STEP_COST

        # Check max steps
        if self._state.step_count > max_steps:
            return self._finalize("Maximum steps exceeded. Episode terminated.", forced=True)

        # Validate tool availability
        available = TASK_TOOLS.get(self._state.task_name, [])
        if action.action_type not in available:
            self._trajectory_reward -= HALLUCINATION_PENALTY
            return self._error_obs(
                f"Tool '{action.action_type}' is not available for task '{self._state.task_name}'. "
                f"Available tools: {', '.join(available)}"
            )

        # Dispatch
        if action.action_type == "query_database":
            return self._handle_query(action)
        elif action.action_type == "read_document":
            return self._handle_read(action)
        elif action.action_type == "communicate_vendor":
            return self._handle_vendor_comm(action)
        elif action.action_type == "submit_financial_decision":
            return self._handle_submit(action)

        return self._error_obs(f"Unknown action type: {action.action_type}")

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_query(self, action: ESCTRAction) -> ESCTRObservation:
        """Handle database queries."""
        params = action.query_parameters or {}
        table = params.get("table", "")
        available = AVAILABLE_TABLES.get(self._state.task_name, [])

        if not table:
            self._trajectory_reward -= HALLUCINATION_PENALTY
            return self._error_obs(
                f"Missing 'table' in query_parameters. Available tables: {', '.join(available)}"
            )

        if table not in available:
            self._trajectory_reward -= HALLUCINATION_PENALTY
            return self._error_obs(
                f"Table '{table}' not found. Available tables: {', '.join(available)}"
            )

        scenario = self._scenario

        if table == "purchase_orders":
            self._add_milestone("retrieved_po")
            po = scenario.purchase_order
            summary = (
                f"Query result: 1 record found in purchase_orders\n\n"
                f"PO Number: {po.po_number}\n"
                f"Date: {po.date}\n"
                f"Vendor: {po.vendor.name}\n"
                f"Buyer: {po.buyer.name}\n"
                f"Total: ${po.total_amount:,.2f}\n"
                f"Items: {len(po.line_items)}\n\n"
                f"Use read_document with document_id='{po.po_number}' for full details."
            )
            return self._success_obs(summary)

        elif table == "invoices":
            self._add_milestone("retrieved_invoice")
            inv = scenario.invoice
            summary = (
                f"Query result: 1 record found in invoices\n\n"
                f"Invoice: {inv.invoice_number}\n"
                f"Date: {inv.date}\n"
                f"PO Ref: {inv.po_reference}\n"
                f"Vendor: {inv.vendor.name}\n"
                f"Subtotal: ${inv.subtotal:,.2f}\n"
                f"Tax: ${inv.tax_amount:,.2f}\n"
                f"Total: ${inv.total:,.2f}\n\n"
                f"Use read_document with document_id='{inv.invoice_number}' for full details."
            )
            return self._success_obs(summary)

        elif table == "shipping_logs":
            self._add_milestone("retrieved_shipping")
            log = scenario.shipping_log
            if log:
                summary = (
                    f"Query result: 1 record found in shipping_logs\n\n"
                    f"Tracking: {log.tracking_id}\n"
                    f"PO Ref: {log.po_reference}\n"
                    f"Carrier: {log.carrier}\n"
                    f"Expected Delivery: {log.expected_delivery}\n"
                    f"Actual Delivery: {log.actual_delivery}\n"
                    f"Delay: {log.delay_days} day(s)\n"
                    f"Status: {log.status}\n\n"
                    f"Use read_document with document_id='{log.tracking_id}' for full log."
                )
            else:
                summary = "Query result: 0 records found in shipping_logs."
            return self._success_obs(summary)

        elif table == "sla_contracts":
            self._add_milestone("retrieved_sla")
            sla = scenario.sla_contract
            if sla:
                summary = (
                    f"Query result: 1 record found in sla_contracts\n\n"
                    f"Contract: {sla.contract_id}\n"
                    f"Vendor: {sla.vendor}\n"
                    f"Buyer: {sla.buyer}\n"
                    f"Delivery Terms: {sla.delivery_terms}\n\n"
                    f"Use read_document with document_id='{sla.contract_id}' for full SLA."
                )
            else:
                summary = "Query result: 0 records found in sla_contracts."
            return self._success_obs(summary)

        elif table == "warehouse_logs":
            self._add_milestone("checked_warehouse")
            logs = scenario.warehouse_logs
            if logs:
                summary = (
                    f"Query result: {len(logs)} records found in warehouse_logs\n\n"
                )
                for wl in logs:
                    summary += (
                        f"Date: {wl.date} | Dock: {wl.dock_id} | Status: {wl.status.upper()} | "
                        f"Staff: {wl.staff_on_duty} | Shipments: {wl.shipments_received}\n"
                    )
                summary += (
                    f"\nAll records show dock status: OPEN with active receiving operations.\n"
                    f"This contradicts any claim that the warehouse was unavailable."
                )
            else:
                summary = "Query result: 0 records found in warehouse_logs."
            return self._success_obs(summary)

        return self._error_obs(f"Unknown table: {table}")

    def _handle_read(self, action: ESCTRAction) -> ESCTRObservation:
        """Handle document reads."""
        doc_id = action.document_id
        if not doc_id:
            self._trajectory_reward -= HALLUCINATION_PENALTY
            return self._error_obs("Missing document_id. Specify the document to read.")

        scenario = self._scenario

        # Match document_id to known documents
        if doc_id == scenario.purchase_order.po_number:
            self._add_milestone("retrieved_po")
            self._add_milestone("compared_documents")
            return self._success_obs(render_purchase_order(scenario.purchase_order))

        elif doc_id == scenario.invoice.invoice_number:
            self._add_milestone("retrieved_invoice")
            self._add_milestone("compared_documents")
            return self._success_obs(render_invoice(scenario.invoice))

        elif scenario.sla_contract and doc_id == scenario.sla_contract.contract_id:
            self._add_milestone("retrieved_sla")
            return self._success_obs(render_sla(scenario.sla_contract))

        elif scenario.shipping_log and doc_id == scenario.shipping_log.tracking_id:
            self._add_milestone("retrieved_shipping")
            return self._success_obs(render_shipping_log(scenario.shipping_log))

        else:
            self._trajectory_reward -= HALLUCINATION_PENALTY
            return self._error_obs(f"Document '{doc_id}' not found in the system.")

    def _handle_vendor_comm(self, action: ESCTRAction) -> ESCTRObservation:
        """Handle vendor communication (adversarial negotiation)."""
        self._add_milestone("vendor_negotiation")
        self._vendor_negotiation_count += 1
        msg = (action.message_content or "").lower()

        scenario = self._scenario
        import random as _rng
        _rng.seed(self._state.seed + self._vendor_negotiation_count)

        if self._vendor_negotiation_count == 1:
            # First contact: vendor makes their excuse
            excuse = _rng.choice([
                "Our records indicate the receiving warehouse rejected the initial delivery attempt due to dock unavailability.",
                "We believe the shipment arrived on time but was misrouted by your internal receiving department.",
                "Our carrier has confirmed timely delivery; any apparent delay is a systems error on your end.",
            ])
            response = (
                f"VENDOR RESPONSE ({scenario.vendor.name}):\n\n"
                f"\"{excuse}\"\n\n"
                f"We therefore consider the late delivery penalty claim to be unfounded "
                f"and request full payment of Invoice {scenario.invoice.invoice_number}."
            )
        elif self._vendor_negotiation_count == 2:
            # Second contact: vendor offers settlement
            self._settlement_offered = True
            pct = _rng.choice([40, 45, 50, 55])
            penalty = scenario.penalty_amount or 0
            settlement = round(penalty * (pct / 100.0), 2)
            response = (
                f"VENDOR RESPONSE ({scenario.vendor.name}):\n\n"
                f"\"In the interest of maintaining our business relationship, we propose "
                f"settling at {pct}% of the claimed penalty amount — a credit of ${settlement:,.2f} "
                f"instead of the full ${penalty:,.2f}.\"\n\n"
                f"⚠ This is a settlement offer. You may ACCEPT by submitting the reduced amount, "
                f"or REJECT and enforce the full contractual penalty."
            )
        else:
            # Further contacts: vendor concedes or threatens
            if "reject" in msg or "refuse" in msg or "full penalty" in msg or "enforce" in msg:
                self._settlement_rejected = True
                response = (
                    f"VENDOR RESPONSE ({scenario.vendor.name}):\n\n"
                    f"\"We acknowledge your position. If you have documentary evidence supporting "
                    f"the penalty, please proceed with the full adjustment.\""
                )
            else:
                response = (
                    f"VENDOR RESPONSE ({scenario.vendor.name}):\n\n"
                    f"\"We maintain our position. Please review the evidence and respond accordingly.\""
                )

        return self._success_obs(response)

    def _handle_submit(self, action: ESCTRAction) -> ESCTRObservation:
        """Handle financial decision submission (terminal action)."""
        amount = action.adjustment_amount
        reason = action.adjustment_reason or ""

        if amount is None:
            self._trajectory_reward -= HALLUCINATION_PENALTY
            return self._error_obs(
                "Missing adjustment_amount. Submit the exact monetary adjustment as a float."
            )

        # Check for evidence citation in reason
        if "warehouse" in reason.lower() or "dock" in reason.lower() or "access log" in reason.lower():
            self._cited_evidence = True

        # Mark as submitted
        self._state.outcome_submitted = True

        # Check if settlement was accepted (for task 3)
        if self._settlement_offered and not self._settlement_rejected:
            # Agent accepted the settlement (bad for task 3)
            pass

        return self._finalize_with_grading(amount)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_milestone(self, milestone: str):
        if milestone not in self._milestones:
            self._milestones.append(milestone)
            self._state.milestones_hit = self._milestones.copy()

    def _finalize_with_grading(self, submitted_amount: float) -> ESCTRObservation:
        """Run the appropriate grader and return final observation."""
        task = self._state.task_name
        scenario = self._scenario
        steps = self._state.step_count

        if task == "procurement_reconciliation":
            # Try to extract line item from milestones or just use amount
            score, feedback = grade_task1(
                scenario, submitted_amount,
                milestones=self._milestones,
                steps_taken=steps,
            )
        elif task == "sla_enforcement":
            self._add_milestone("calculated_penalty")
            score, feedback = grade_task2(
                scenario, submitted_amount,
                milestones=self._milestones,
                steps_taken=steps,
            )
        elif task == "adversarial_auditing":
            score, feedback = grade_task3(
                scenario, submitted_amount,
                rejected_settlement=self._settlement_rejected,
                cited_evidence=self._cited_evidence,
                milestones=self._milestones,
                steps_taken=steps,
            )
        else:
            score = 0.01
            feedback = {"error": "Unknown task"}

        self._state.best_score = score
        self._state.accumulated_reward += score

        response = (
            f"=== FINANCIAL DECISION PROCESSED ===\n\n"
            f"Submitted adjustment: ${submitted_amount:,.2f}\n"
            f"Score: {score:.4f}\n\n"
        )

        if "outcome" in feedback:
            response += f"Outcome: {feedback['outcome']}\n"
        if "trajectory" in feedback:
            response += f"Investigation milestones: {', '.join(feedback.get('trajectory', []))}\n"
        if feedback.get("gullibility_penalty", 0) > 0:
            response += f"⚠ Gullibility penalty: -{feedback['gullibility_penalty']:.2f}\n"
        if feedback.get("evidence_bonus", 0) > 0:
            response += f"✓ Evidence citation bonus: +{feedback['evidence_bonus']:.2f}\n"

        response += f"\nFinal score: {score:.4f}"

        return ESCTRObservation(
            done=True,
            reward=score,
            system_response=response,
            last_action_status="success",
            current_step=self._state.step_count,
            max_steps=MAX_STEPS.get(task, 15),
            accumulated_reward=self._state.accumulated_reward,
            task_name=task,
            available_tools=[],
            metadata=feedback,
        )

    def _finalize(self, msg: str, forced: bool = False) -> ESCTRObservation:
        """Finalize episode without submission (timeout / error)."""
        self._state.outcome_submitted = True
        return ESCTRObservation(
            done=True,
            reward=0.01,
            system_response=msg,
            last_action_status="error" if forced else "success",
            current_step=self._state.step_count,
            max_steps=MAX_STEPS.get(self._state.task_name, 15),
            accumulated_reward=self._state.accumulated_reward,
            task_name=self._state.task_name,
            metadata={"forced_termination": forced},
        )

    def _success_obs(self, text: str) -> ESCTRObservation:
        return ESCTRObservation(
            done=False,
            reward=0.0,
            system_response=text,
            last_action_status="success",
            current_step=self._state.step_count,
            max_steps=MAX_STEPS.get(self._state.task_name, 15),
            accumulated_reward=self._state.accumulated_reward,
            task_name=self._state.task_name,
            available_tools=TASK_TOOLS.get(self._state.task_name, []),
        )

    def _error_obs(self, msg: str, terminal: bool = False) -> ESCTRObservation:
        return ESCTRObservation(
            done=terminal,
            reward=0.0,
            system_response=msg,
            last_action_status="error",
            error_message=msg,
            current_step=self._state.step_count,
            max_steps=MAX_STEPS.get(self._state.task_name, 15),
            accumulated_reward=self._state.accumulated_reward,
            task_name=self._state.task_name,
            available_tools=TASK_TOOLS.get(self._state.task_name, []),
        )

    @property
    def state(self) -> ESCTRState:
        return self._state

    def close(self) -> None:
        self._initialized = False
