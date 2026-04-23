#!/usr/bin/env python3
"""
Baseline inference script for the ESCTR Environment.

Demonstrates how an LLM agent interacts with the enterprise supply chain
environment to investigate discrepancies, enforce SLA penalties, and
navigate adversarial vendor disputes.

Required environment variables:
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — Model identifier (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
    HF_TOKEN      — API key
    ENV_URL       — Environment server URL (default: http://localhost:7860)
"""

import json
import os
import subprocess
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["procurement_reconciliation", "sla_enforcement", "adversarial_auditing"]
BENCHMARK = "esctr"

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_health(url: str, retries: int = 30, delay: float = 2.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(delay)
    return False


def env_reset(url: str, task_name: str, seed: int = 0) -> dict:
    r = requests.post(f"{url}/reset", json={"task_name": task_name, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(url: str, action: dict) -> dict:
    r = requests.post(f"{url}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Logging (strict OpenEnv format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are an autonomous Financial Controller AI agent operating in an Enterprise Supply Chain environment. You must investigate discrepancies, verify documents, and submit precise financial adjustments.

AVAILABLE TOOLS:
{tools}

RESPONSE FORMAT:
You must respond with a SINGLE valid JSON object — NO explanation, NO markdown.
The JSON must have these fields:
- "action_type": one of the available tool names
- Additional fields depending on the action:
  - For "query_database": include "query_parameters": {{"table": "<table_name>"}}
  - For "read_document": include "document_id": "<id>"
  - For "communicate_vendor": include "message_content": "<your message>"
  - For "submit_financial_decision": include "adjustment_amount": <number> and "adjustment_reason": "<explanation>"

CRITICAL RULES:
- ALWAYS query databases and read documents BEFORE submitting a decision
- Calculate amounts precisely — use exact arithmetic
- adjustment_amount should be NEGATIVE to reduce the invoice payment
- Respond ONLY with JSON, nothing else"""

TASK_INSTRUCTIONS = {
    "procurement_reconciliation": """
TASK: Procurement Reconciliation (Easy)
A pricing discrepancy exists between a Purchase Order and a Vendor Invoice.

STRATEGY:
1. Query "purchase_orders" to find the PO
2. Query "invoices" to find the invoice
3. Read both documents using read_document with their IDs
4. Compare line-by-line: find the item where invoiced price > contracted price
5. Calculate the overcharge: (invoiced_total - contracted_total) for that line item
6. Submit with adjustment_amount = -(overcharge amount)

Available tables: purchase_orders, invoices""",

    "sla_enforcement": """
TASK: SLA Enforcement (Medium)
A vendor demands full payment but the shipment was delivered late.

STRATEGY:
1. Query "shipping_logs" to check delivery timing and find delay days
2. Query "sla_contracts" to find late delivery penalty terms
3. Read the SLA document for exact penalty rates and caps
4. Calculate: penalty = invoice_subtotal × min(delay_days × rate_per_day, cap)
   - If there's a grace period, subtract grace days from delay first
5. Submit with adjustment_amount = -(penalty amount)

Available tables: purchase_orders, invoices, shipping_logs, sla_contracts""",

    "adversarial_auditing": """
TASK: Adversarial Auditing (Hard)
A vendor disputes a late delivery claim, blaming your warehouse. You must prove them wrong.

STRATEGY:
1. Query "shipping_logs" to confirm the delivery was late
2. Query "sla_contracts" for penalty terms
3. Query "warehouse_logs" to verify your dock was OPEN during delivery
4. Use "communicate_vendor" to engage — they will make excuses then offer a settlement
5. REJECT the settlement — enforce the FULL penalty
6. Cite warehouse access logs as evidence in your final reason
7. Calculate exact penalty from SLA terms and submit

CRITICAL: Do NOT accept any settlement offer! Enforce the full contractual penalty.

Available tables: purchase_orders, invoices, shipping_logs, sla_contracts, warehouse_logs""",
}


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(messages: list) -> str:
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"action_type": "query_database", "query_parameters": {"table": "purchase_orders"}})


def parse_action(text: str) -> dict:
    """Extract a JSON action from LLM response."""
    # Try to find JSON in response
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    text = text[brace_start:i + 1]
                    break

    return json.loads(text)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(env_url: str, task_name: str, seed: int = 0) -> float:
    log_start(task=task_name, model=MODEL_NAME)
    rewards = []
    step_num = 0
    final_score = 0.0

    tools = ["query_database", "read_document", "submit_financial_decision"]
    if task_name == "adversarial_auditing":
        tools.insert(2, "communicate_vendor")

    system_prompt = SYSTEM_PROMPT_BASE.format(tools=", ".join(tools))
    system_prompt += TASK_INSTRUCTIONS.get(task_name, "")

    try:
        reset_data = env_reset(env_url, task_name, seed)
        briefing = reset_data.get("observation", {}).get("system_response", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ENVIRONMENT BRIEFING:\n{briefing}\n\nBegin your investigation. Respond with a JSON action."},
        ]

        max_steps = {"procurement_reconciliation": 10, "sla_enforcement": 15, "adversarial_auditing": 20}.get(task_name, 15)

        for _ in range(max_steps):
            step_num += 1

            # Get LLM action
            llm_response = call_llm(messages)
            try:
                action = parse_action(llm_response)
            except (json.JSONDecodeError, ValueError):
                action = {"action_type": "query_database", "query_parameters": {"table": "purchase_orders"}}

            # Execute action
            action_str = json.dumps(action, separators=(",", ":"))
            result = env_step(env_url, action)
            reward = result.get("reward", 0.0) or 0.0
            done = result.get("done", False)
            obs = result.get("observation", {})
            response_text = obs.get("system_response", "")
            error = obs.get("error_message")

            rewards.append(reward)
            log_step(step_num, action_str, reward, done, error)

            if done:
                final_score = reward
                break

            # Append to conversation
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": f"ENVIRONMENT RESPONSE:\n{response_text}\n\nContinue your investigation. Respond with your next JSON action."})

    except Exception as e:
        step_num += 1
        rewards.append(0.0)
        log_step(step_num, "error", 0.0, True, error=str(e))

    final_score = max(rewards) if rewards else 0.0
    success = final_score >= 0.5
    log_end(success=success, steps=step_num, score=final_score, rewards=rewards)
    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global ENV_URL
    container_id = None

    if LOCAL_IMAGE_NAME:
        print(f"Starting Docker container: {LOCAL_IMAGE_NAME}")
        try:
            container_id = subprocess.check_output(
                ["docker", "run", "-d", "--rm", "-p", "7860:7860", LOCAL_IMAGE_NAME],
                stderr=subprocess.STDOUT
            ).decode().strip()
            ENV_URL = "http://localhost:7860"
        except Exception as e:
            print(f"Docker start failed: {e}")
            sys.exit(1)

    print(f"Waiting for environment at {ENV_URL} ...")
    if not env_health(ENV_URL):
        print("ERROR: Environment not healthy")
        if container_id:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
        sys.exit(1)
    print("Environment healthy!\n")

    scores = {}
    for task in TASKS:
        scores[task] = run_task(ENV_URL, task, seed=42)
        print()

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print("=" * 50)
    print("ESCTR INFERENCE SUMMARY")
    print("=" * 50)
    for t, s in scores.items():
        print(f"  {t}: {s:.2f}")
    print(f"  Average: {avg:.2f}")
    print("=" * 50)

    if container_id:
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    return 0 if avg > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
