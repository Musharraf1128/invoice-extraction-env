#!/usr/bin/env python3
"""
Baseline inference script for the Invoice Extraction Environment.

This script demonstrates how an LLM agent interacts with the environment
to extract structured data from invoice documents. It runs all five tasks
(simple_invoice, messy_invoice, multi_document, corrupted_scan, adversarial_invoice)
and logs results in the mandatory OpenEnv [START]/[STEP]/[END] format.

Required environment variables:
    API_BASE_URL       — OpenAI-compatible API endpoint
    MODEL_NAME         — Model identifier (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
    HF_TOKEN           — API key / Hugging Face token (no default)
    ENV_URL            — URL of the running environment server (default: http://localhost:7860)
    LOCAL_IMAGE_NAME   — (Optional) Docker image name for from_docker_image() style
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

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["simple_invoice", "messy_invoice", "multi_document", "corrupted_scan", "adversarial_invoice"]

# ---------------------------------------------------------------------------
# LLM Client (OpenAI-compatible, configured via env vars)
# ---------------------------------------------------------------------------
llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ---------------------------------------------------------------------------
# Environment helpers (stateful HTTP)
# ---------------------------------------------------------------------------

def env_health(url: str, retries: int = 30, delay: float = 2.0) -> bool:
    """Wait for the environment server to be healthy."""
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
    """Reset the environment for a given task."""
    r = requests.post(
        f"{url}/reset",
        json={"task_name": task_name, "seed": seed},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(url: str, command: str, payload: str = "") -> dict:
    """Execute a step in the environment."""
    r = requests.post(
        f"{url}/step",
        json={"action": {"command": command, "payload": payload}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You are an expert data extraction assistant. Given the following document text, extract the specified fields and return ONLY a valid JSON object.

DOCUMENT:
{document}

REQUIRED FIELDS:
{fields}

RULES:
- Return ONLY a valid JSON object, no explanation or markdown
- For dates, use YYYY-MM-DD format (e.g. 2024-01-15)
- For monetary amounts, use plain numbers without currency symbols (e.g. 1134.00)
- For line_items, use an array of objects with keys: description, quantity, unit_price, amount
- If a field cannot be found, use null
{task_specific_rules}

IMPORTANT: Ensure your extracted subtotal + tax = total. Verify math consistency.

JSON:"""

TASK_RULES = {
    "simple_invoice": "",
    "messy_invoice": (
        "- This document uses informal formatting, abbreviations, and shorthand\n"
        "- Look past formatting irregularities to find the actual values\n"
        "- 'subtot', 's/t', 'sub' = subtotal; 'tx' = tax; 'amt due' = total"
    ),
    "multi_document": (
        "- This contains MULTIPLE document sections (PO, Invoice, Credit Memo, etc.)\n"
        "- Extract from the INVOICE section primarily\n"
        "- adjusted_total is the final amount after credits/payments\n"
        "- po_number is the purchase order reference number\n"
        "- adjustment_reason describes why the total was adjusted"
    ),
    "corrupted_scan": (
        "- WARNING: This is an OCR-scanned document with character errors\n"
        "- Common OCR substitutions: 0<->O, 1<->l<->I, 5<->S, 8<->B\n"
        "- Mentally correct OCR errors to recover the true values\n"
        "- 'lNV' = 'INV', 'S' in numbers = '5', 'O' in numbers = '0'\n"
        "- Verify all numbers by cross-checking (qty * unit_price = amount)"
    ),
    "adversarial_invoice": (
        "- CAUTION: This document contains DECOY fields and contradictions\n"
        "- Multiple invoice numbers may appear — use the CURRENT/ACTIVE one, not voided/draft ones\n"
        "- If there is a reissue date, use that as the date (not the original date)\n"
        "- subtotal is the ADJUSTED subtotal after any discounts\n"
        "- discount_amount is the monetary discount value\n"
        "- original_total is what the total WOULD have been without adjustments\n"
        "- discrepancy_notes: describe ALL discrepancies, adjustments, and calculations\n"
        "- po_number: the purchase order reference if present, else null\n"
        "- Cross-reference different sections to find contradictions"
    ),
}

REFINE_PROMPT = """You previously extracted data from an invoice but some fields were incorrect.

DOCUMENT:
{document}

YOUR PREVIOUS EXTRACTION:
{previous}

FIELDS NEEDING IMPROVEMENT: {weak_fields}

FEEDBACK:
{feedback}

Please re-extract ALL fields and return ONLY a valid JSON object with corrections.
Pay special attention to the fields listed above.

RULES:
- Return ONLY a valid JSON object, no explanation or markdown
- For dates, use YYYY-MM-DD format
- For monetary amounts, use plain numbers without currency symbols
- For line_items, use an array of objects with keys: description, quantity, unit_price, amount
- If a field cannot be determined, use null
- VERIFY: subtotal + tax should equal total
{task_specific_rules}

JSON:"""


def call_llm(prompt: str) -> str:
    """Call the LLM and return the response text."""
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"error": str(e)})


def extract_json_from_response(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try to find JSON object directly
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start : i + 1]

    return text


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_task(env_url: str, task_name: str, seed: int = 0) -> float:
    """Run a single task and return the final score."""
    print(f"[START] task={task_name} env=invoice-extraction model={MODEL_NAME}")

    rewards = []
    step_num = 0
    final_score = 0.0

    try:
        # Reset environment
        reset_result = env_reset(env_url, task_name, seed=seed)

        # Step 1: View the document
        step_num += 1
        result = env_step(env_url, "view_document")
        document_text = result.get("observation", {}).get("text", "")
        reward = result.get("reward", 0.0) or 0.0
        done = result.get("done", False)
        rewards.append(reward)
        print(
            f"[STEP] step={step_num} action=view_document() "
            f"reward={reward:.2f} done={str(done).lower()} error=null"
        )

        # Step 2: View required fields
        step_num += 1
        result = env_step(env_url, "view_fields")
        fields_text = result.get("observation", {}).get("text", "")
        required_fields = result.get("observation", {}).get("required_fields", [])
        reward = result.get("reward", 0.0) or 0.0
        done = result.get("done", False)
        rewards.append(reward)
        print(
            f"[STEP] step={step_num} action=view_fields() "
            f"reward={reward:.2f} done={str(done).lower()} error=null"
        )

        # Step 3: Use LLM to extract fields
        fields_str = "\n".join(f"- {f}" for f in required_fields)
        task_rules = TASK_RULES.get(task_name, "")
        prompt = EXTRACT_PROMPT.format(
            document=document_text,
            fields=fields_str,
            task_specific_rules=task_rules,
        )
        llm_response = call_llm(prompt)
        extracted_json = extract_json_from_response(llm_response)

        # Step 4: Submit extraction
        step_num += 1
        result = env_step(env_url, "extract", extracted_json)
        reward = result.get("reward", 0.0) or 0.0
        done = result.get("done", False)
        obs = result.get("observation", {})
        rewards.append(reward)
        print(
            f"[STEP] step={step_num} action=submit_extraction() "
            f"reward={reward:.2f} done={str(done).lower()} error=null"
        )

        final_score = reward

        # If not done and score < 0.9, try to refine
        if not done and reward < 0.9:
            # Get feedback
            step_num += 1
            fb_result = env_step(env_url, "get_feedback")
            feedback_text = fb_result.get("observation", {}).get("text", "")
            fb_reward = fb_result.get("reward", 0.0) or 0.0
            rewards.append(fb_reward)
            print(
                f"[STEP] step={step_num} action=get_feedback() "
                f"reward={fb_reward:.2f} done=false error=null"
            )

            # Identify weak fields from metadata
            field_scores = obs.get("metadata", {}).get("field_scores", {})
            weak_fields = [f for f, s in field_scores.items() if s < 0.8]

            # Refine with LLM
            task_rules = TASK_RULES.get(task_name, "")
            refine_prompt = REFINE_PROMPT.format(
                document=document_text,
                previous=extracted_json,
                weak_fields=", ".join(weak_fields) if weak_fields else "all fields",
                feedback=feedback_text,
                task_specific_rules=task_rules,
            )
            refined_response = call_llm(refine_prompt)
            refined_json = extract_json_from_response(refined_response)

            # Submit refined extraction
            step_num += 1
            result2 = env_step(env_url, "extract", refined_json)
            reward2 = result2.get("reward", 0.0) or 0.0
            done = result2.get("done", False)
            rewards.append(reward2)
            print(
                f"[STEP] step={step_num} action=submit_refined_extraction() "
                f"reward={reward2:.2f} done={str(done).lower()} error=null"
            )

            final_score = max(final_score, reward2)

    except Exception as e:
        step_num += 1
        rewards.append(0.0)
        print(
            f"[STEP] step={step_num} action=error "
            f"reward=0.00 done=true error={str(e)}"
        )

    # Use best score across all attempts
    final_score = max(rewards) if rewards else 0.0
    success = final_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )

    return final_score


def main():
    global ENV_URL

    container_id = None

    # If LOCAL_IMAGE_NAME is set, start a Docker container
    if LOCAL_IMAGE_NAME:
        print(f"Starting Docker container from image: {LOCAL_IMAGE_NAME}")
        try:
            container_id = subprocess.check_output(
                ["docker", "run", "-d", "--rm", "-p", "7860:7860", LOCAL_IMAGE_NAME],
                stderr=subprocess.STDOUT,
            ).decode().strip()
            ENV_URL = "http://localhost:7860"
            print(f"Container started: {container_id[:12]}")
        except Exception as e:
            print(f"Failed to start Docker container: {e}")
            sys.exit(1)

    # Wait for environment to be healthy
    print(f"Waiting for environment at {ENV_URL} ...")
    if not env_health(ENV_URL):
        print("ERROR: Environment failed to become healthy")
        if container_id:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
        sys.exit(1)
    print("Environment is healthy!\n")

    # Run all tasks
    scores = {}
    for task_name in TASKS:
        score = run_task(ENV_URL, task_name, seed=42)
        scores[task_name] = score
        print()  # Blank line between tasks

    # Summary
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for task, score in scores.items():
        print(f"  {task}: {score:.2f}")
    print(f"  Average: {avg_score:.2f}")
    print("=" * 50)

    # Cleanup Docker container
    if container_id:
        print(f"Stopping container {container_id[:12]} ...")
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    return 0 if avg_score > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
