"""
Interactive Gradio UI for the ESCTR Environment.

Lets judges and users play the environment directly in the browser:
- Pick a task and seed
- Use tools step by step
- See the investigation log and reward in real-time
"""

import gradio as gr
import random
from .environment import ESCTREnvironment
from .models import ESCTRAction


# ── Theming ──────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.esctr-header { text-align: center; margin-bottom: 1rem; }
.esctr-header h1 { font-size: 2rem; font-weight: 700; }
.esctr-header p { color: #6b7280; font-size: 1rem; }
.log-box { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.reward-display { font-size: 2.5rem; font-weight: 800; text-align: center; }
.tool-btn { min-height: 48px; }
"""


# ── State management ─────────────────────────────────────────────────────────

def create_env():
    return ESCTREnvironment()


def reset_episode(task_name, seed_text):
    """Reset the environment with a task and seed."""
    env = create_env()
    seed = int(seed_text) if seed_text.strip() else random.randint(0, 99999)
    obs = env.reset(task_name=task_name, seed=seed)

    log = f"{'='*60}\n"
    log += f"  🏢 ESCTR — New Episode\n"
    log += f"  Task: {task_name} | Seed: {seed}\n"
    log += f"{'='*60}\n\n"
    log += f"📋 BRIEFING:\n{obs.system_response}\n\n"
    log += f"{'─'*60}\n"

    status = f"⏳ Step 0 | Reward: 0.00 | Status: Investigating..."

    return (
        env,           # state: env object
        log,           # log textbox
        "0.00",        # reward display
        status,        # status bar
        str(seed),     # seed display
        0,             # step counter
        gr.update(interactive=True),  # enable tool buttons
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def execute_tool(env, log, step_count, action_type, **kwargs):
    """Execute a tool action and update the log."""
    if env is None:
        return env, log + "\n⚠️ Please reset the environment first!\n", "0.00", "Not started", step_count

    try:
        action = ESCTRAction(action_type=action_type, **kwargs)
        obs = env.step(action)
    except Exception as e:
        log += f"\n❌ ERROR: {str(e)}\n"
        return env, log, "0.00", "Error", step_count

    step_count += 1
    reward = obs.reward
    done = obs.done

    # Format the tool call
    param_str = ", ".join(f'{k}="{v}"' for k, v in kwargs.items() if v)
    log += f"\n🔧 Step {step_count}: {action_type}({param_str})\n"
    log += f"{'─'*40}\n"

    # Truncate very long responses for readability
    response = obs.system_response
    if len(response) > 1500:
        response = response[:1500] + "\n... [truncated for display]"
    log += f"{response}\n"
    log += f"{'─'*40}\n"

    if done:
        log += f"\n{'='*60}\n"
        log += f"  ✅ EPISODE COMPLETE\n"
        log += f"  Final Reward: {reward:.4f}\n"
        log += f"  Steps Used: {step_count}\n"
        log += f"{'='*60}\n"
        status = f"✅ Done in {step_count} steps | Final Reward: {reward:.4f}"
    else:
        status = f"⏳ Step {step_count} | Reward: {reward:.4f} | Investigating..."

    reward_str = f"{reward:.4f}"

    return env, log, reward_str, status, step_count


def query_db(env, log, step_count, table):
    if not table:
        log += "\n⚠️ Please select a table to query.\n"
        return env, log, "0.00", "Select a table", step_count
    return execute_tool(env, log, step_count, "query_database", query_parameters={"table": table})


def read_doc(env, log, step_count, doc_id):
    if not doc_id.strip():
        log += "\n⚠️ Please enter a document ID (e.g., PO-2025-1234).\n"
        return env, log, "0.00", "Enter a document ID", step_count
    return execute_tool(env, log, step_count, "read_document", document_id=doc_id.strip())


def contact_vendor(env, log, step_count, message):
    if not message.strip():
        log += "\n⚠️ Please enter a message for the vendor.\n"
        return env, log, "0.00", "Enter a message", step_count
    return execute_tool(env, log, step_count, "communicate_vendor", message_content=message.strip())


def submit_decision(env, log, step_count, amount, reason):
    try:
        amt = float(amount)
    except (ValueError, TypeError):
        log += "\n⚠️ Please enter a valid numeric amount.\n"
        return env, log, "0.00", "Enter valid amount", step_count
    if not reason.strip():
        reason = "Financial adjustment based on investigation"
    return execute_tool(env, log, step_count, "submit_financial_decision",
                        adjustment_amount=amt, adjustment_reason=reason.strip())


# ── Build UI ─────────────────────────────────────────────────────────────────

def build_gradio_app():
    with gr.Blocks(theme=THEME, css=CSS, title="ESCTR Environment") as demo:

        # Hidden state
        env_state = gr.State(None)
        step_counter = gr.State(0)

        # ── Header ───────────────────────────────────────────────
        gr.HTML("""
        <div class="esctr-header">
            <h1>🏢 ESCTR: Enterprise Supply Chain & Tax Reconciliation</h1>
            <p>Training LLMs as autonomous financial auditors — powered by RLVR</p>
            <p style="font-size: 0.85rem; color: #9ca3af;">
                OpenEnv Hackathon 2026 · 
                <a href="https://github.com/Musharraf1128/esctr-environment" target="_blank">GitHub</a> · 
                <a href="https://huggingface.co/spaces/musharraf7/esctr-grpo-trained" target="_blank">Training Dashboard</a>
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left: Controls ───────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Episode Controls")

                task_dropdown = gr.Dropdown(
                    choices=[
                        ("🟢 Procurement Reconciliation (Easy)", "procurement_reconciliation"),
                        ("🟡 SLA Enforcement (Medium)", "sla_enforcement"),
                        ("🔴 Adversarial Auditing (Hard)", "adversarial_auditing"),
                    ],
                    value="procurement_reconciliation",
                    label="Task",
                )
                seed_input = gr.Textbox(
                    label="Seed (leave empty for random)",
                    placeholder="e.g., 42",
                    value="",
                )
                reset_btn = gr.Button("🔄 Start New Episode", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🔧 Tools")

                # Tool 1: Query Database
                with gr.Group():
                    gr.Markdown("**📊 Query Database**")
                    db_table = gr.Dropdown(
                        choices=["purchase_orders", "invoices", "shipping_logs", "sla_contracts", "warehouse_logs"],
                        label="Table",
                        value="purchase_orders",
                    )
                    query_btn = gr.Button("Run Query", elem_classes="tool-btn")

                # Tool 2: Read Document
                with gr.Group():
                    gr.Markdown("**📄 Read Document**")
                    doc_id_input = gr.Textbox(label="Document ID", placeholder="PO-2025-1234 or INV-2025-5678")
                    read_btn = gr.Button("Read Document", elem_classes="tool-btn")

                # Tool 3: Contact Vendor
                with gr.Group():
                    gr.Markdown("**💬 Contact Vendor**")
                    vendor_msg = gr.Textbox(label="Message", placeholder="We reject your settlement...", lines=2)
                    vendor_btn = gr.Button("Send Message", elem_classes="tool-btn")

                # Tool 4: Submit Decision
                with gr.Group():
                    gr.Markdown("**⚖️ Submit Financial Decision**")
                    adj_amount = gr.Textbox(label="Adjustment Amount ($)", placeholder="e.g., -450.00")
                    adj_reason = gr.Textbox(label="Reason", placeholder="Overcharge on line item...", lines=2)
                    submit_btn = gr.Button("Submit Decision", variant="stop", elem_classes="tool-btn")

            # ── Right: Log & Results ─────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Investigation Log")
                status_bar = gr.Textbox(label="Status", value="Click 'Start New Episode' to begin", interactive=False)

                with gr.Row():
                    reward_display = gr.Textbox(
                        label="💰 Current Reward",
                        value="—",
                        interactive=False,
                        elem_classes="reward-display",
                    )
                    seed_display = gr.Textbox(label="🎲 Seed", value="—", interactive=False)

                log_output = gr.Textbox(
                    label="Full Investigation Log",
                    value="Waiting for episode to start...",
                    lines=25,
                    max_lines=50,
                    interactive=False,
                    elem_classes="log-box",
                )

        # ── Event Handlers ───────────────────────────────────────

        reset_outputs = [
            env_state, log_output, reward_display, status_bar,
            seed_display, step_counter,
            query_btn, read_btn, vendor_btn, submit_btn,
        ]
        reset_btn.click(
            fn=reset_episode,
            inputs=[task_dropdown, seed_input],
            outputs=reset_outputs,
        )

        tool_outputs = [env_state, log_output, reward_display, status_bar, step_counter]

        query_btn.click(
            fn=query_db,
            inputs=[env_state, log_output, step_counter, db_table],
            outputs=tool_outputs,
        )
        read_btn.click(
            fn=read_doc,
            inputs=[env_state, log_output, step_counter, doc_id_input],
            outputs=tool_outputs,
        )
        vendor_btn.click(
            fn=contact_vendor,
            inputs=[env_state, log_output, step_counter, vendor_msg],
            outputs=tool_outputs,
        )
        submit_btn.click(
            fn=submit_decision,
            inputs=[env_state, log_output, step_counter, adj_amount, adj_reason],
            outputs=tool_outputs,
        )

    return demo
