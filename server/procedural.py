"""
Procedural Document Generation Engine.

Generates infinite invoice variations using seed-based randomization.
Addresses the "data sparsity" critique by providing virtually unlimited
training configurations while maintaining deterministic reproducibility.
"""

import random
import string
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data pools for procedural generation
# ---------------------------------------------------------------------------

VENDOR_POOL = [
    ("Acme Corporation", "123 Business Avenue", "New York", "NY", "10001"),
    ("TechStart Solutions LLC", "890 Innovation Drive, Suite 200", "San Francisco", "CA", "94105"),
    ("Global Supplies Inc.", "2500 Industrial Parkway", "Detroit", "MI", "48201"),
    ("Pinnacle Systems Ltd.", "77 Summit Road", "Boston", "MA", "02101"),
    ("Nexus Digital Services", "400 Cloud Way", "Seattle", "WA", "98101"),
    ("Ironclad Manufacturing Co.", "1200 Forge Lane", "Pittsburgh", "PA", "15201"),
    ("Brightwave Analytics", "55 Data Drive", "Austin", "TX", "78701"),
    ("SilverLine Logistics", "909 Transport Blvd", "Memphis", "TN", "38101"),
    ("Quantum Computing Corp.", "1 Qubit Plaza", "Boulder", "CO", "80301"),
    ("Evergreen Office Supplies", "330 Elm Street", "Portland", "OR", "97201"),
    ("Atlas Engineering Group", "620 Blueprint Ave", "Houston", "TX", "77001"),
    ("Cobalt Healthcare Solutions", "88 Wellness Pkwy", "Minneapolis", "MN", "55401"),
    ("Meridian Consulting Partners", "250 Strategy Lane", "Chicago", "IL", "60601"),
    ("Vanguard Robotics Inc.", "15 Automation Circle", "San Jose", "CA", "95101"),
    ("Horizon Energy Systems", "700 Solar Way", "Denver", "CO", "80201"),
]

CUSTOMER_POOL = [
    ("Widget Co.", "456 Commerce Street", "Chicago", "IL", "60601"),
    ("DataFlow Inc.", "321 Analytics Blvd", "Austin", "TX", "78701"),
    ("Riverside Manufacturing", "780 Factory Road", "Cleveland", "OH", "44101"),
    ("Summit Enterprises", "100 Peak Drive", "Denver", "CO", "80201"),
    ("Cascade Solutions Group", "55 River Bend Rd", "Portland", "OR", "97201"),
    ("Sterling Financial Corp.", "800 Wall St", "New York", "NY", "10005"),
    ("Bluestone Retail Inc.", "120 Market Square", "Philadelphia", "PA", "19101"),
    ("Northstar Logistics", "450 Freight Way", "Minneapolis", "MN", "55401"),
    ("Pacific Tech Ventures", "700 Bay Ave", "San Diego", "CA", "92101"),
    ("Redwood Construction LLC", "35 Builder Lane", "Sacramento", "CA", "95801"),
    ("Falcon Aerospace", "1 Launchpad Dr", "Huntsville", "AL", "35801"),
    ("Cedar Health Systems", "200 Wellness Blvd", "Nashville", "TN", "37201"),
    ("Granite Insurance Group", "90 Coverage Ct", "Hartford", "CT", "06101"),
    ("Oakmont Education Trust", "60 Campus Way", "Ann Arbor", "MI", "48101"),
    ("Sapphire Media Holdings", "500 Broadcast Pl", "Los Angeles", "CA", "90001"),
]

PRODUCT_CATALOG = [
    # (description, min_price, max_price, unit)
    ("Widget Type A", 15.00, 50.00, "unit"),
    ("Widget Type B", 25.00, 80.00, "unit"),
    ("Consulting Service", 50.00, 200.00, "hour"),
    ("Cloud Hosting (Monthly)", 200.00, 800.00, "month"),
    ("API Integration Setup", 500.00, 3000.00, "unit"),
    ("Technical Support", 60.00, 150.00, "hour"),
    ("Steel Bolts (Box/100)", 8.00, 20.00, "box"),
    ("Copper Wire (500ft Roll)", 50.00, 120.00, "roll"),
    ("Safety Goggles (Pack/10)", 20.00, 60.00, "pack"),
    ("Welding Rods (Bundle)", 15.00, 40.00, "bundle"),
    ("Software License (Annual)", 100.00, 2000.00, "license"),
    ("Office Furniture Set", 200.00, 800.00, "set"),
    ("Network Switch (24-port)", 150.00, 500.00, "unit"),
    ("Printer Ink Cartridge", 20.00, 80.00, "unit"),
    ("Industrial Adhesive (Gallon)", 25.00, 75.00, "gallon"),
    ("LED Panel Light", 30.00, 100.00, "unit"),
    ("HVAC Filter (Pack/4)", 15.00, 45.00, "pack"),
    ("Hydraulic Pump Assembly", 300.00, 1200.00, "unit"),
    ("Precision Bearing Set", 40.00, 150.00, "set"),
    ("Thermal Insulation Roll", 60.00, 200.00, "roll"),
    ("Data Backup Service", 75.00, 300.00, "month"),
    ("Security Audit", 500.00, 2500.00, "audit"),
    ("Custom Report Development", 200.00, 1000.00, "report"),
    ("Training Workshop", 150.00, 500.00, "session"),
    ("Prototype Fabrication", 1000.00, 5000.00, "unit"),
]

TAX_RATES = [0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10]

OCR_SUBSTITUTIONS = {
    "O": "0", "0": "O", "l": "1", "1": "l", "I": "l",
    "S": "5", "5": "S", "B": "8", "8": "B", "m": "rn",
    "a": "o", "e": "c", "n": "ri",
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class ProceduralEngine:
    """Seed-based procedural document generator."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def _pick(self, pool: list) -> Any:
        return self.rng.choice(pool)

    def _gen_invoice_number(self, prefix: str = "") -> str:
        if not prefix:
            prefix = self.rng.choice(["INV", "TS", "GS", "NX", "PC", "BW", "SL", "QC"])
        year = self.rng.choice([2023, 2024, 2025])
        num = self.rng.randint(1, 9999)
        fmt = self.rng.choice([
            f"{prefix}-{year}-{num:04d}",
            f"{prefix}{num:04d}",
            f"{prefix}-{num:04d}-{self.rng.choice(['A','B','R1','R2'])}",
        ])
        return fmt

    def _gen_date(self) -> Tuple[str, str]:
        """Returns (display_date, normalized YYYY-MM-DD)."""
        year = self.rng.choice([2023, 2024, 2025])
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        norm = f"{year}-{month:02d}-{day:02d}"
        fmt_choice = self.rng.randint(0, 3)
        if fmt_choice == 0:
            display = f"{MONTHS[month-1]} {day}, {year}"
        elif fmt_choice == 1:
            display = f"{month:02d}/{day:02d}/{year}"
        elif fmt_choice == 2:
            display = f"{day}-{MONTHS[month-1][:3]}-{year}"
        else:
            display = norm
        return display, norm

    def _gen_line_items(self, count: int = 0) -> List[Dict[str, Any]]:
        if count == 0:
            count = self.rng.randint(2, 6)
        products = self.rng.sample(PRODUCT_CATALOG, min(count, len(PRODUCT_CATALOG)))
        items = []
        for desc, min_p, max_p, _unit in products:
            qty = self.rng.randint(1, 50)
            price = round(self.rng.uniform(min_p, max_p), 2)
            amount = round(qty * price, 2)
            items.append({
                "description": desc,
                "quantity": qty,
                "unit_price": price,
                "amount": amount,
            })
        return items

    def generate_simple(self) -> Dict[str, Any]:
        vendor = self._pick(VENDOR_POOL)
        customer = self._pick(CUSTOMER_POOL)
        inv_num = self._gen_invoice_number()
        display_date, norm_date = self._gen_date()
        items = self._gen_line_items()
        subtotal = round(sum(i["amount"] for i in items), 2)
        tax_rate = self._pick(TAX_RATES)
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)
        tax_pct = int(tax_rate * 100) if tax_rate * 100 == int(tax_rate * 100) else tax_rate * 100

        items_text = ""
        for it in items:
            items_text += f"{it['description']:<30s} {it['quantity']:>5d}    ${it['unit_price']:>10.2f}   ${it['amount']:>10.2f}\n"

        text = f"""INVOICE

Invoice Number: {inv_num}
Date: {display_date}

From:
  {vendor[0]}
  {vendor[1]}
  {vendor[2]}, {vendor[3]} {vendor[4]}

Bill To:
  {customer[0]}
  {customer[1]}
  {customer[2]}, {customer[3]} {customer[4]}

Description                    Qty    Unit Price      Amount
---------------------------------------------------------------
{items_text}
                                      Subtotal:  ${subtotal:,.2f}
                                      Tax ({tax_pct}%):   ${tax:,.2f}
                                      Total:     ${total:,.2f}

Payment Terms: Net {self.rng.choice([15, 30, 45, 60])}
"""
        ground_truth = {
            "invoice_number": inv_num,
            "date": norm_date,
            "vendor_name": vendor[0],
            "customer_name": customer[0],
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "line_items": items,
        }
        return {"id": f"proc_simple_{self.rng.randint(1000,9999)}", "text": text, "ground_truth": ground_truth}

    def generate_messy(self) -> Dict[str, Any]:
        base = self.generate_simple()
        gt = base["ground_truth"]
        vendor = gt["vendor_name"]
        customer = gt["customer_name"]
        items = gt["line_items"]

        abbrevs = {"Subtotal": self._pick(["subtot", "s/t", "sub"]),
                    "Tax": self._pick(["tx", "tax", "vat"]),
                    "Total": self._pick(["TOTAL DUE", "amt due", "grand total", "balance"])}

        items_text = ""
        for it in items:
            desc_short = it["description"].split("(")[0].strip().lower()
            qty = it["quantity"]
            price = it["unit_price"]
            amt = it["amount"]
            fmt = self.rng.choice([
                f"{qty}x {desc_short} @ {price:.0f}           {amt:.0f}",
                f"{desc_short} -- {qty} @ {price:.2f} ea ........... {amt:.0f}",
                f"{desc_short}...${amt:.0f}",
            ])
            items_text += fmt + "\n"

        text = f"""{vendor.lower()}
{self._pick(VENDOR_POOL)[2].lower()}, {self._pick(VENDOR_POOL)[3]}

inv# {gt['invoice_number']}
dt: {gt['date']}

cust: {customer.split('.')[0].split(',')[0].lower()}

-- charges --
{items_text}
{abbrevs['Subtotal']}: ${gt['subtotal']:.0f}
{abbrevs['Tax']}: {gt['tax']:.2f}
========
{abbrevs['Total']} ${gt['total']:,.2f}

pay within 30 days
"""
        return {"id": f"proc_messy_{self.rng.randint(1000,9999)}", "text": text, "ground_truth": gt}

    def _apply_ocr_corruption(self, text: str, intensity: float = 0.15) -> str:
        result = list(text)
        for i, ch in enumerate(result):
            if ch in OCR_SUBSTITUTIONS and self.rng.random() < intensity:
                result[i] = OCR_SUBSTITUTIONS[ch]
        return "".join(result)

    def generate_corrupted(self) -> Dict[str, Any]:
        base = self.generate_simple()
        corrupted_text = self._apply_ocr_corruption(base["text"], 0.18)
        header = self._pick([
            "SC4NNED D0CUMENT - Page 1 of 1\n\n",
            "[SCAN QUALITY: P00R - SOME CHARACTERS MAY BE lNCORRECT]\n\n",
            "---FAXED DOCUMENT---\nQUALITY: [####===---] 40%\n\n",
        ])
        footer = self._pick([
            "\n\n--- END 0F SCAN ---",
            "\n\n[PAGE 1/1 - SCAN C0MPLETE]",
            "\n\n---END FAX---",
        ])
        return {
            "id": f"proc_corrupt_{self.rng.randint(1000,9999)}",
            "text": header + corrupted_text + footer,
            "ground_truth": base["ground_truth"],
        }

    def generate_multi_document(self) -> Dict[str, Any]:
        base = self.generate_simple()
        gt = base["ground_truth"]
        po_num = f"PO-{self.rng.choice(['A','B','C','D'])}-{self.rng.randint(2024,2025)}-{self.rng.randint(100,999)}"
        po_date_display, _po_norm = self._gen_date()

        items_po = ""
        for it in gt["line_items"]:
            items_po += f"- {it['quantity']}x {it['description']} @ ${it['unit_price']:.2f} = ${it['amount']:.2f}\n"

        credit_amount = round(self._pick(gt["line_items"])["unit_price"] * self.rng.randint(1, 3), 2)
        credit_tax = round(credit_amount * 0.07, 2)
        credit_total = round(credit_amount + credit_tax, 2)
        adjusted_total = round(gt["total"] - credit_total, 2)
        reason = self._pick([
            "Defective items returned",
            "Partial delivery — remaining items backordered",
            "Pricing error on original invoice",
            "Duplicate charge for services",
        ])

        text = f"""=== PURCHASE ORDER ===
PO Number: {po_num}
Date: {po_date_display}
Vendor: {gt['vendor_name']}
Buyer: {gt['customer_name']}

Ordered Items:
{items_po}
PO Total: ${gt['subtotal']:,.2f} (before tax)

=== INVOICE ===
{base['text']}
Reference PO: {po_num}

=== CREDIT MEMO ===
Credit Memo #: CM-{self.rng.randint(2024,2025)}-{self.rng.randint(100,999)}
Reference Invoice: {gt['invoice_number']}
Reason: {reason}
Credit Amount: ${credit_amount:.2f}
Tax Adjustment: ${credit_tax:.2f}
Total Credit: -${credit_total:.2f}

=== SUMMARY ===
Original Invoice: ${gt['total']:,.2f}
Credit Applied: -${credit_total:.2f}
Adjusted Balance Due: ${adjusted_total:,.2f}
"""
        gt_multi = dict(gt)
        gt_multi["po_number"] = po_num
        gt_multi["adjustment_reason"] = reason
        gt_multi["adjusted_total"] = adjusted_total
        return {"id": f"proc_multi_{self.rng.randint(1000,9999)}", "text": text, "ground_truth": gt_multi}

    def generate_adversarial(self) -> Dict[str, Any]:
        base = self.generate_simple()
        gt = base["ground_truth"]
        original_subtotal = gt["subtotal"]
        discount_pct = self._pick([0.05, 0.08, 0.10, 0.12, 0.15])
        discount_amount = round(original_subtotal * discount_pct, 2)
        adjusted_subtotal = round(original_subtotal - discount_amount, 2)
        tax_rate = self._pick(TAX_RATES)
        new_tax = round(adjusted_subtotal * tax_rate, 2)
        new_total = round(adjusted_subtotal + new_tax, 2)
        old_tax = round(original_subtotal * tax_rate, 2)
        original_total = round(original_subtotal + old_tax, 2)

        draft_inv = f"DRAFT-INV-{self.rng.randint(100,999)}"
        real_inv = gt["invoice_number"] + self._pick(["-R2", "-FINAL", "-REV1"])
        po_num = f"PO-{self.rng.randint(2024,2025)}-{self.rng.randint(100,999)}"
        _, reissue_date = self._gen_date()
        tax_pct = int(tax_rate * 100) if tax_rate * 100 == int(tax_rate * 100) else round(tax_rate * 100, 1)

        items_text = ""
        for it in gt["line_items"]:
            items_text += f"{it['description']:<30s} {it['quantity']:>5d}    ${it['unit_price']:>10.2f}   ${it['amount']:>10.2f}\n"

        discount_pct_display = int(discount_pct * 100) if discount_pct * 100 == int(discount_pct * 100) else round(discount_pct * 100, 1)

        text = f"""INVOICE

*** IMPORTANT: This replaces previous invoice {draft_inv} which was voided ***

Invoice Number: {real_inv}
Previous Reference: {draft_inv} (VOIDED — DO NOT USE)
Date: {gt['date']}
Reissue Date: {reissue_date}
PO Reference: {po_num}

From:
  {gt['vendor_name']}

Bill To:
  {gt['customer_name']}

Description                    Qty    Unit Price      Amount
---------------------------------------------------------------
{items_text}  ** EARLY PAYMENT DISCOUNT: -{discount_pct_display}% applied **

                                      Subtotal:    ${original_subtotal:,.2f}
                                 Discount ({discount_pct_display}%):  -${discount_amount:,.2f}
                            Adjusted Subtotal:   ${adjusted_subtotal:,.2f}
                                      Tax ({tax_pct}%):    ${new_tax:,.2f}
                                      Total:       ${new_total:,.2f}

NOTE: Original quote was ${original_total:,.2f} but discount applied.

!!! BUDGET VARIANCE ALERT !!!
PO Authorized: ${original_subtotal:,.2f}
Actual (pre-tax): ${adjusted_subtotal:,.2f}
Variance: -${discount_amount:,.2f} UNDER BUDGET

Payment Terms: Net 10 (discounted) / Net 30 (full price ${original_total:,.2f})
"""
        discrepancy = (
            f"{discount_pct_display}% early payment discount applied. "
            f"Reissued invoice replaces voided {draft_inv}. "
            f"Adjusted subtotal ${adjusted_subtotal:,.2f} vs original ${original_subtotal:,.2f}."
        )

        gt_adv = {
            "invoice_number": real_inv,
            "date": reissue_date,
            "vendor_name": gt["vendor_name"],
            "customer_name": gt["customer_name"],
            "subtotal": adjusted_subtotal,
            "tax": new_tax,
            "total": new_total,
            "line_items": gt["line_items"],
            "po_number": po_num,
            "discount_amount": discount_amount,
            "original_total": original_total,
            "discrepancy_notes": discrepancy,
        }
        return {"id": f"proc_adv_{self.rng.randint(1000,9999)}", "text": text, "ground_truth": gt_adv}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GENERATORS = {
    "simple_invoice": "generate_simple",
    "messy_invoice": "generate_messy",
    "multi_document": "generate_multi_document",
    "corrupted_scan": "generate_corrupted",
    "adversarial_invoice": "generate_adversarial",
}


def generate_document(task_name: str, seed: int = 0) -> Dict[str, Any]:
    """Generate a procedural document for the given task and seed."""
    engine = ProceduralEngine(seed)
    method = GENERATORS.get(task_name, "generate_simple")
    return getattr(engine, method)()
