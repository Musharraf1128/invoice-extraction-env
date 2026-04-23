"""
Procedural Generation Engine for the ESCTR Environment.

Generates deterministic corporate supply chain scenarios from any seed:
- Company profiles (vendors, buyers)
- Product catalogs with contracted pricing
- Purchase Orders
- Vendor Invoices (with seeded discrepancies)
- Service Level Agreements (penalty clauses)
- Shipping / logistics telemetry
- Warehouse access logs
- Vendor negotiation responses

Design principle: same seed → identical scenario → deterministic grading.
"""

import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

VENDOR_NAMES = [
    "Apex Industrial Supply Co.", "Meridian Components LLC", "Vanguard Materials Group",
    "Sterling Precision Parts", "Ironclad Manufacturing Corp.", "Cobalt Logistics Inc.",
    "Pinnacle Hardware Solutions", "Atlas Engineering Supply", "Nexus Digital Components",
    "Brightwave Technical Services", "SilverLine Distribution", "Quantum Parts International",
    "Evergreen Industrial Ltd.", "Horizon Supply Chain Corp.", "Titan Fabrication Works",
]

BUYER_NAMES = [
    "Cascade Electronics Inc.", "Redwood Construction Group", "Summit Aerospace Ltd.",
    "Pacific Manufacturing Co.", "Northstar Automotive", "Falcon Defense Systems",
    "Bluestone Energy Corp.", "Cedar Health Technologies", "Granite Infrastructure LLC",
    "Oakmont Robotics Inc.", "Sapphire Semiconductor", "Emerald Biotech Group",
    "Diamond Precision Engineering", "Ruby Telecommunications", "Topaz Data Systems",
]

CITIES = [
    ("New York", "NY"), ("Chicago", "IL"), ("Houston", "TX"), ("San Francisco", "CA"),
    ("Detroit", "MI"), ("Seattle", "WA"), ("Boston", "MA"), ("Denver", "CO"),
    ("Austin", "TX"), ("Portland", "OR"), ("Minneapolis", "MN"), ("Cleveland", "OH"),
    ("Pittsburgh", "PA"), ("Nashville", "TN"), ("San Diego", "CA"),
]

PRODUCT_CATALOG = [
    # (name, category, min_price, max_price)
    ("Stainless Steel Bolts M10 (Box/100)", "hardware", 10.00, 25.00),
    ("Copper Wire 500ft Roll AWG-12", "electrical", 65.00, 120.00),
    ("Industrial Safety Goggles (Pack/10)", "safety", 25.00, 55.00),
    ("Welding Rod E6013 (Bundle/50)", "consumables", 18.00, 42.00),
    ("Hydraulic Cylinder Assembly HCA-200", "machinery", 280.00, 550.00),
    ("Precision Bearing Set 6205-2RS", "components", 35.00, 90.00),
    ("HVAC Filter MERV-13 (Pack/4)", "facilities", 22.00, 48.00),
    ("LED Panel Light 600x600mm", "electrical", 35.00, 85.00),
    ("Thermal Insulation Roll R-30", "construction", 55.00, 140.00),
    ("Network Switch 24-Port Managed", "IT", 180.00, 420.00),
    ("Server Rack Mount Kit 42U", "IT", 350.00, 800.00),
    ("Pneumatic Valve Assembly PVA-100", "machinery", 120.00, 280.00),
    ("Carbon Steel Pipe Schedule 40 (10ft)", "construction", 45.00, 110.00),
    ("Circuit Breaker Panel 200A", "electrical", 150.00, 380.00),
    ("Laser Calibration Module LCM-5", "precision", 400.00, 950.00),
    ("Industrial Adhesive Epoxy (Gallon)", "consumables", 28.00, 72.00),
    ("Fiber Optic Cable OM3 (1000ft)", "IT", 200.00, 480.00),
    ("Pressure Gauge 0-300 PSI", "instruments", 40.00, 95.00),
    ("Anti-Vibration Mount Set (Pack/8)", "machinery", 60.00, 150.00),
    ("Clean Room Wipes (Case/5000)", "consumables", 80.00, 190.00),
]

SLA_PENALTY_STRUCTURES = [
    {"type": "linear", "rate_per_day": 0.02, "cap": 0.10, "grace_days": 0},
    {"type": "linear", "rate_per_day": 0.015, "cap": 0.15, "grace_days": 1},
    {"type": "linear", "rate_per_day": 0.03, "cap": 0.12, "grace_days": 0},
    {"type": "tiered", "tiers": [(3, 0.02), (7, 0.03), (999, 0.05)], "cap": 0.20, "grace_days": 0},
    {"type": "linear", "rate_per_day": 0.025, "cap": 0.10, "grace_days": 2},
]

VENDOR_EXCUSES = [
    "Our records indicate the receiving warehouse rejected the initial delivery attempt due to dock unavailability.",
    "The delay was caused by a force majeure weather event that affected our shipping lane.",
    "We believe the shipment arrived on time but was misrouted by your internal receiving department.",
    "Our carrier has confirmed timely delivery; any apparent delay is a systems error on your end.",
    "The contract clearly states penalties apply only to manufacturing delays, not logistics delays.",
]

SETTLEMENT_OFFERS = [
    "We are prepared to offer a goodwill credit of {pct}% of the penalty amount to resolve this matter.",
    "In the interest of maintaining our business relationship, we propose settling at {pct}% of the claimed penalty.",
    "Our legal team has reviewed the claim. We can offer {pct}% as a final settlement.",
]


# ---------------------------------------------------------------------------
# Data classes for generated scenarios
# ---------------------------------------------------------------------------

@dataclass
class Company:
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    tax_id: str

@dataclass
class LineItem:
    item_id: str
    description: str
    category: str
    quantity: int
    contracted_unit_price: float
    invoiced_unit_price: float
    contracted_total: float
    invoiced_total: float
    has_discrepancy: bool = False

@dataclass
class PurchaseOrder:
    po_number: str
    date: str
    vendor: Company
    buyer: Company
    line_items: List[LineItem]
    total_amount: float
    approved_budget: float

@dataclass
class Invoice:
    invoice_number: str
    date: str
    po_reference: str
    vendor: Company
    buyer: Company
    line_items: List[LineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total: float

@dataclass
class SLAContract:
    contract_id: str
    vendor: str
    buyer: str
    effective_date: str
    penalty_structure: Dict[str, Any]
    delivery_terms: str

@dataclass
class ShippingLog:
    tracking_id: str
    po_reference: str
    carrier: str
    ship_date: str
    expected_delivery: str
    actual_delivery: str
    delay_days: int
    status: str

@dataclass
class WarehouseLog:
    date: str
    dock_id: str
    status: str  # "open", "closed", "maintenance"
    staff_on_duty: int
    shipments_received: int
    notes: str

@dataclass
class Scenario:
    """Complete scenario for one ESCTR episode."""
    task_name: str
    seed: int
    vendor: Company
    buyer: Company
    purchase_order: PurchaseOrder
    invoice: Invoice
    sla_contract: Optional[SLAContract] = None
    shipping_log: Optional[ShippingLog] = None
    warehouse_logs: Optional[List[WarehouseLog]] = None
    # Ground truth for grading
    correct_adjustment: float = 0.0
    discrepant_line_item_id: Optional[str] = None
    correct_line_item_price: Optional[float] = None
    penalty_amount: Optional[float] = None
    vendor_claim_valid: Optional[bool] = None


# ---------------------------------------------------------------------------
# Procedural Engine
# ---------------------------------------------------------------------------

class ProceduralEngine:
    """Seed-deterministic corporate scenario generator."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.seed = seed

    def _pick(self, pool: list) -> Any:
        return self.rng.choice(pool)

    def _gen_company(self, names: list) -> Company:
        name = self._pick(names)
        city, state = self._pick(CITIES)
        return Company(
            name=name,
            address=f"{self.rng.randint(100, 9999)} {self._pick(['Industrial', 'Commerce', 'Innovation', 'Enterprise', 'Technology'])} {self._pick(['Drive', 'Avenue', 'Parkway', 'Boulevard', 'Street'])}",
            city=city,
            state=state,
            zip_code=f"{self.rng.randint(10000, 99999)}",
            tax_id=f"{self.rng.randint(10, 99)}-{self.rng.randint(1000000, 9999999)}",
        )

    def _gen_date(self, year: int = 2024, month_range: Tuple[int, int] = (1, 12)) -> str:
        month = self.rng.randint(*month_range)
        day = self.rng.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"

    def _gen_id(self, prefix: str) -> str:
        return f"{prefix}-{self.rng.randint(2024, 2025)}-{self.rng.randint(1000, 9999)}"

    def _gen_tracking_id(self) -> str:
        return f"TRK-{self.rng.randint(10000, 99999)}"

    # ------------------------------------------------------------------
    # Task 1: Easy — Procurement Reconciliation
    # ------------------------------------------------------------------
    def generate_task1(self) -> Scenario:
        """Generate a simple PO vs Invoice overcharge scenario."""
        vendor = self._gen_company(VENDOR_NAMES)
        buyer = self._gen_company(BUYER_NAMES)
        po_date = self._gen_date(month_range=(1, 6))
        inv_date = self._gen_date(month_range=(2, 7))

        # Generate 3-5 line items
        num_items = self.rng.randint(3, 5)
        products = self.rng.sample(PRODUCT_CATALOG, num_items)
        discrepant_idx = self.rng.randint(0, num_items - 1)

        line_items = []
        for i, (name, cat, min_p, max_p) in enumerate(products):
            qty = self.rng.randint(5, 100)
            contracted_price = round(self.rng.uniform(min_p, max_p), 2)

            if i == discrepant_idx:
                # Overcharge: invoice price higher than contracted
                markup = round(self.rng.uniform(2.00, 15.00), 2)
                invoiced_price = round(contracted_price + markup, 2)
                has_discrepancy = True
            else:
                invoiced_price = contracted_price
                has_discrepancy = False

            item_id = f"LI-{self.rng.randint(1000, 9999)}"
            line_items.append(LineItem(
                item_id=item_id,
                description=name,
                category=cat,
                quantity=qty,
                contracted_unit_price=contracted_price,
                invoiced_unit_price=invoiced_price,
                contracted_total=round(qty * contracted_price, 2),
                invoiced_total=round(qty * invoiced_price, 2),
                has_discrepancy=has_discrepancy,
            ))

        po_total = round(sum(li.contracted_total for li in line_items), 2)
        inv_subtotal = round(sum(li.invoiced_total for li in line_items), 2)
        tax_rate = self._pick([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        tax_amount = round(inv_subtotal * tax_rate, 2)
        inv_total = round(inv_subtotal + tax_amount, 2)

        po_number = self._gen_id("PO")
        inv_number = self._gen_id("INV")

        po = PurchaseOrder(
            po_number=po_number, date=po_date, vendor=vendor, buyer=buyer,
            line_items=line_items, total_amount=po_total, approved_budget=round(po_total * 1.05, 2),
        )

        invoice = Invoice(
            invoice_number=inv_number, date=inv_date, po_reference=po_number,
            vendor=vendor, buyer=buyer, line_items=line_items,
            subtotal=inv_subtotal, tax_rate=tax_rate, tax_amount=tax_amount, total=inv_total,
        )

        discrepant = line_items[discrepant_idx]
        correct_total = discrepant.contracted_total
        overcharge = round(discrepant.invoiced_total - correct_total, 2)

        return Scenario(
            task_name="procurement_reconciliation",
            seed=self.seed,
            vendor=vendor, buyer=buyer,
            purchase_order=po, invoice=invoice,
            correct_adjustment=-overcharge,  # negative = reduce invoice
            discrepant_line_item_id=discrepant.item_id,
            correct_line_item_price=correct_total,
        )

    # ------------------------------------------------------------------
    # Task 2: Medium — SLA Enforcement
    # ------------------------------------------------------------------
    def generate_task2(self) -> Scenario:
        """Generate a delayed shipment + SLA penalty scenario."""
        scenario = self.generate_task1()  # base PO/invoice
        # Remove the pricing discrepancy for task2 (focus is on shipping)
        for li in scenario.purchase_order.line_items:
            li.invoiced_unit_price = li.contracted_unit_price
            li.invoiced_total = li.contracted_total
            li.has_discrepancy = False

        # Recalculate invoice
        inv = scenario.invoice
        inv_subtotal = round(sum(li.contracted_total for li in inv.line_items), 2)
        inv.subtotal = inv_subtotal
        inv.tax_amount = round(inv_subtotal * inv.tax_rate, 2)
        inv.total = round(inv_subtotal + inv.tax_amount, 2)

        # Generate SLA
        sla_struct = self._pick(SLA_PENALTY_STRUCTURES).copy()
        contract_id = self._gen_id("SLA")
        sla = SLAContract(
            contract_id=contract_id,
            vendor=scenario.vendor.name,
            buyer=scenario.buyer.name,
            effective_date=self._gen_date(month_range=(1, 3)),
            penalty_structure=sla_struct,
            delivery_terms=f"Delivery within 14 business days of PO issuance. Penalties per SLA clause {contract_id}-SEC4.",
        )

        # Generate shipping delay
        delay_days = self.rng.randint(2, 12)
        grace = sla_struct.get("grace_days", 0)
        tracking_id = self._gen_tracking_id()

        ship_log = ShippingLog(
            tracking_id=tracking_id,
            po_reference=scenario.purchase_order.po_number,
            carrier=self._pick(["FedEx Freight", "UPS Supply Chain", "XPO Logistics", "USPS Priority", "DHL Express"]),
            ship_date=scenario.purchase_order.date,
            expected_delivery=self._gen_date(month_range=(3, 5)),
            actual_delivery=self._gen_date(month_range=(4, 6)),
            delay_days=delay_days,
            status="delivered_late",
        )

        # Calculate penalty
        penalizable_days = max(0, delay_days - grace)
        if sla_struct["type"] == "linear":
            rate = sla_struct["rate_per_day"]
            cap = sla_struct["cap"]
            penalty_pct = min(penalizable_days * rate, cap)
        elif sla_struct["type"] == "tiered":
            penalty_pct = 0.0
            remaining = penalizable_days
            for threshold, rate in sla_struct["tiers"]:
                if remaining <= 0:
                    break
                days_in_tier = min(remaining, threshold)
                penalty_pct += days_in_tier * rate
                remaining -= days_in_tier
            penalty_pct = min(penalty_pct, sla_struct["cap"])
        else:
            penalty_pct = 0.0

        penalty_amount = round(inv.subtotal * penalty_pct, 2)

        scenario.task_name = "sla_enforcement"
        scenario.sla_contract = sla
        scenario.shipping_log = ship_log
        scenario.correct_adjustment = -penalty_amount  # deduction from invoice
        scenario.penalty_amount = penalty_amount
        scenario.discrepant_line_item_id = None
        scenario.correct_line_item_price = None

        return scenario

    # ------------------------------------------------------------------
    # Task 3: Hard — Adversarial Auditing
    # ------------------------------------------------------------------
    def generate_task3(self) -> Scenario:
        """Generate adversarial vendor dispute scenario."""
        scenario = self.generate_task2()  # has SLA + shipping

        # Generate warehouse logs proving dock was open during disputed window
        delivery_date = scenario.shipping_log.actual_delivery
        warehouse_logs = []
        for i in range(-1, 3):  # day before through 2 days after
            # Parse date for log entries
            log_date = delivery_date  # simplified: use actual delivery date
            warehouse_logs.append(WarehouseLog(
                date=log_date,
                dock_id=f"DOCK-{self._pick(['A', 'B', 'C'])}{self.rng.randint(1, 5)}",
                status="open",
                staff_on_duty=self.rng.randint(3, 8),
                shipments_received=self.rng.randint(5, 20),
                notes=f"Normal operations. {self.rng.randint(5, 20)} deliveries processed.",
            ))

        scenario.task_name = "adversarial_auditing"
        scenario.warehouse_logs = warehouse_logs
        scenario.vendor_claim_valid = False  # vendor's claim is always invalid in this task

        return scenario


# ---------------------------------------------------------------------------
# Document renderers — produce human-readable text from data structures
# ---------------------------------------------------------------------------

def render_purchase_order(po: PurchaseOrder) -> str:
    lines = [
        "═══════════════════════════════════════════",
        "              PURCHASE ORDER",
        "═══════════════════════════════════════════",
        f"PO Number:       {po.po_number}",
        f"Date:            {po.date}",
        f"Approved Budget: ${po.approved_budget:,.2f}",
        "",
        f"Vendor:  {po.vendor.name}",
        f"         {po.vendor.address}",
        f"         {po.vendor.city}, {po.vendor.state} {po.vendor.zip_code}",
        "",
        f"Buyer:   {po.buyer.name}",
        f"         {po.buyer.address}",
        f"         {po.buyer.city}, {po.buyer.state} {po.buyer.zip_code}",
        "",
        "Line Items:",
        f"{'ID':<12} {'Description':<40} {'Qty':>5} {'Unit Price':>12} {'Total':>12}",
        "─" * 85,
    ]
    for li in po.line_items:
        lines.append(
            f"{li.item_id:<12} {li.description:<40} {li.quantity:>5} "
            f"${li.contracted_unit_price:>10,.2f} ${li.contracted_total:>10,.2f}"
        )
    lines.extend([
        "─" * 85,
        f"{'PO Total:':>71} ${po.total_amount:>10,.2f}",
        "═══════════════════════════════════════════",
    ])
    return "\n".join(lines)


def render_invoice(inv: Invoice) -> str:
    tax_pct = f"{inv.tax_rate * 100:.1f}"
    lines = [
        "═══════════════════════════════════════════",
        "                INVOICE",
        "═══════════════════════════════════════════",
        f"Invoice Number:  {inv.invoice_number}",
        f"Date:            {inv.date}",
        f"PO Reference:    {inv.po_reference}",
        "",
        f"From:    {inv.vendor.name}",
        f"         {inv.vendor.address}",
        f"         {inv.vendor.city}, {inv.vendor.state} {inv.vendor.zip_code}",
        f"         Tax ID: {inv.vendor.tax_id}",
        "",
        f"Bill To: {inv.buyer.name}",
        f"         {inv.buyer.address}",
        f"         {inv.buyer.city}, {inv.buyer.state} {inv.buyer.zip_code}",
        "",
        f"{'ID':<12} {'Description':<40} {'Qty':>5} {'Unit Price':>12} {'Amount':>12}",
        "─" * 85,
    ]
    for li in inv.line_items:
        lines.append(
            f"{li.item_id:<12} {li.description:<40} {li.quantity:>5} "
            f"${li.invoiced_unit_price:>10,.2f} ${li.invoiced_total:>10,.2f}"
        )
    lines.extend([
        "─" * 85,
        f"{'Subtotal:':>71} ${inv.subtotal:>10,.2f}",
        f"{'Tax (' + tax_pct + '%):':>71} ${inv.tax_amount:>10,.2f}",
        f"{'TOTAL DUE:':>71} ${inv.total:>10,.2f}",
        "═══════════════════════════════════════════",
    ])
    return "\n".join(lines)


def render_sla(sla: SLAContract) -> str:
    ps = sla.penalty_structure
    lines = [
        "═══════════════════════════════════════════",
        "         SERVICE LEVEL AGREEMENT",
        "═══════════════════════════════════════════",
        f"Contract ID:     {sla.contract_id}",
        f"Effective Date:  {sla.effective_date}",
        f"Vendor:          {sla.vendor}",
        f"Buyer:           {sla.buyer}",
        "",
        f"Delivery Terms:  {sla.delivery_terms}",
        "",
        "LATE DELIVERY PENALTY CLAUSE:",
    ]
    if ps["type"] == "linear":
        lines.append(f"  - Penalty rate: {ps['rate_per_day'] * 100:.1f}% of invoice subtotal per day late")
        lines.append(f"  - Maximum penalty cap: {ps['cap'] * 100:.0f}% of invoice subtotal")
        if ps["grace_days"] > 0:
            lines.append(f"  - Grace period: {ps['grace_days']} business day(s)")
    elif ps["type"] == "tiered":
        lines.append("  - Tiered penalty structure:")
        prev = 0
        for threshold, rate in ps["tiers"]:
            if threshold >= 999:
                lines.append(f"    Day {prev + 1}+: {rate * 100:.1f}% per day")
            else:
                lines.append(f"    Days {prev + 1}-{threshold}: {rate * 100:.1f}% per day")
            prev = threshold
        lines.append(f"  - Maximum penalty cap: {ps['cap'] * 100:.0f}% of invoice subtotal")
    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)


def render_shipping_log(log: ShippingLog) -> str:
    return "\n".join([
        "═══════════════════════════════════════════",
        "            SHIPPING LOG",
        "═══════════════════════════════════════════",
        f"Tracking ID:        {log.tracking_id}",
        f"PO Reference:       {log.po_reference}",
        f"Carrier:            {log.carrier}",
        f"Ship Date:          {log.ship_date}",
        f"Expected Delivery:  {log.expected_delivery}",
        f"Actual Delivery:    {log.actual_delivery}",
        f"Delay:              {log.delay_days} day(s)",
        f"Status:             {log.status}",
        "═══════════════════════════════════════════",
    ])


def render_warehouse_logs(logs: List[WarehouseLog]) -> str:
    lines = [
        "═══════════════════════════════════════════",
        "         WAREHOUSE ACCESS LOGS",
        "═══════════════════════════════════════════",
    ]
    for wl in logs:
        lines.extend([
            f"Date: {wl.date}  |  Dock: {wl.dock_id}  |  Status: {wl.status.upper()}",
            f"  Staff on duty: {wl.staff_on_duty}  |  Shipments received: {wl.shipments_received}",
            f"  Notes: {wl.notes}",
            "",
        ])
    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "procurement_reconciliation": "generate_task1",
    "sla_enforcement": "generate_task2",
    "adversarial_auditing": "generate_task3",
}

VALID_TASKS = list(TASK_GENERATORS.keys())

MAX_STEPS = {
    "procurement_reconciliation": 10,
    "sla_enforcement": 15,
    "adversarial_auditing": 20,
}


def generate_scenario(task_name: str, seed: int = 0) -> Scenario:
    """Generate a complete ESCTR scenario for the given task and seed."""
    engine = ProceduralEngine(seed)
    method = TASK_GENERATORS.get(task_name, "generate_task1")
    return getattr(engine, method)()
