"""
Document corpus for the Invoice Extraction Environment.

Contains synthetic but realistic invoice/receipt documents across 3 difficulty levels.
Each document has raw text and ground truth extraction targets.
"""

DOCUMENTS = {
    # =========================================================================
    # SIMPLE INVOICES — Clean formatting, clear labels, consistent structure
    # =========================================================================
    "simple_invoice": [
        {
            "id": "simple_001",
            "text": """INVOICE

Invoice Number: INV-2024-001
Date: January 15, 2024

From:
  Acme Corporation
  123 Business Avenue
  New York, NY 10001

Bill To:
  Widget Co.
  456 Commerce Street
  Chicago, IL 60601

Description                Qty    Unit Price    Amount
---------------------------------------------------------
Widget Type A               10      $25.00     $250.00
Widget Type B                5      $40.00     $200.00
Consulting Service           8      $75.00     $600.00

                                   Subtotal:  $1,050.00
                                   Tax (8%):     $84.00
                                   Total:     $1,134.00

Payment Terms: Net 30
""",
            "ground_truth": {
                "invoice_number": "INV-2024-001",
                "date": "2024-01-15",
                "vendor_name": "Acme Corporation",
                "customer_name": "Widget Co.",
                "subtotal": 1050.00,
                "tax": 84.00,
                "total": 1134.00,
                "line_items": [
                    {"description": "Widget Type A", "quantity": 10, "unit_price": 25.00, "amount": 250.00},
                    {"description": "Widget Type B", "quantity": 5, "unit_price": 40.00, "amount": 200.00},
                    {"description": "Consulting Service", "quantity": 8, "unit_price": 75.00, "amount": 600.00},
                ],
            },
        },
        {
            "id": "simple_002",
            "text": """INVOICE

Invoice #: TS-5892
Invoice Date: March 3, 2024

Vendor:
  TechStart Solutions LLC
  890 Innovation Drive, Suite 200
  San Francisco, CA 94105

Customer:
  DataFlow Inc.
  321 Analytics Blvd
  Austin, TX 78701

Item                          Qty   Unit Price     Total
----------------------------------------------------------
Cloud Hosting (Monthly)         1     $450.00    $450.00
API Integration Setup           1   $1,200.00  $1,200.00
Technical Support (hours)      12      $95.00  $1,140.00

                                    Subtotal:  $2,790.00
                                    Tax (7%):    $195.30
                                    Total:     $2,985.30

Due Date: April 2, 2024
""",
            "ground_truth": {
                "invoice_number": "TS-5892",
                "date": "2024-03-03",
                "vendor_name": "TechStart Solutions LLC",
                "customer_name": "DataFlow Inc.",
                "subtotal": 2790.00,
                "tax": 195.30,
                "total": 2985.30,
                "line_items": [
                    {"description": "Cloud Hosting (Monthly)", "quantity": 1, "unit_price": 450.00, "amount": 450.00},
                    {"description": "API Integration Setup", "quantity": 1, "unit_price": 1200.00, "amount": 1200.00},
                    {"description": "Technical Support (hours)", "quantity": 12, "unit_price": 95.00, "amount": 1140.00},
                ],
            },
        },
        {
            "id": "simple_003",
            "text": """INVOICE

Invoice Number: GS-2024-0147
Date: February 20, 2024

From:
  Global Supplies Inc.
  2500 Industrial Parkway
  Detroit, MI 48201

To:
  Riverside Manufacturing
  780 Factory Road
  Cleveland, OH 44101

Product                    Qty    Price Each    Line Total
-----------------------------------------------------------
Steel Bolts (Box/100)       50       $12.50       $625.00
Copper Wire (500ft Roll)     8       $85.00       $680.00
Safety Goggles (Pack/10)    20       $35.00       $700.00
Welding Rods (Bundle)       15       $22.00       $330.00

                                    Subtotal:   $2,335.00
                                    Sales Tax:    $163.45
                                    Invoice Total: $2,498.45

Terms: Net 45
""",
            "ground_truth": {
                "invoice_number": "GS-2024-0147",
                "date": "2024-02-20",
                "vendor_name": "Global Supplies Inc.",
                "customer_name": "Riverside Manufacturing",
                "subtotal": 2335.00,
                "tax": 163.45,
                "total": 2498.45,
                "line_items": [
                    {"description": "Steel Bolts (Box/100)", "quantity": 50, "unit_price": 12.50, "amount": 625.00},
                    {"description": "Copper Wire (500ft Roll)", "quantity": 8, "unit_price": 85.00, "amount": 680.00},
                    {"description": "Safety Goggles (Pack/10)", "quantity": 20, "unit_price": 35.00, "amount": 700.00},
                    {"description": "Welding Rods (Bundle)", "quantity": 15, "unit_price": 22.00, "amount": 330.00},
                ],
            },
        },
    ],

    # =========================================================================
    # MESSY INVOICES — Inconsistent formatting, abbreviations, typos
    # =========================================================================
    "messy_invoice": [
        {
            "id": "messy_001",
            "text": """ACME Corp
123 Biz Ave., NY 10001
Ph: (212) 555-0100

inv# ACM-987
dt: Jan 15 '24

BILL TO:
widgetco / 456 commerce, chicago il

---items---
10x WidgetA @ 25           250
5x WidgetB @ 40            200
8hrs consulting @75/hr     600
                          ------
                    subtot 1050
                    tx 8%:   84
              TOTAL DUE: $1,134

pay within 30 days
""",
            "ground_truth": {
                "invoice_number": "ACM-987",
                "date": "2024-01-15",
                "vendor_name": "ACME Corp",
                "customer_name": "widgetco",
                "subtotal": 1050.00,
                "tax": 84.00,
                "total": 1134.00,
                "line_items": [
                    {"description": "WidgetA", "quantity": 10, "unit_price": 25.00, "amount": 250.00},
                    {"description": "WidgetB", "quantity": 5, "unit_price": 40.00, "amount": 200.00},
                    {"description": "consulting", "quantity": 8, "unit_price": 75.00, "amount": 600.00},
                ],
            },
        },
        {
            "id": "messy_002",
            "text": """techstart solutions
san fran, CA

INVOICE  ts5892-b
date 03/03/2024

cust: DataFlow
      austin TX

-- charges --
hosting (cloud, monthly plan)...$450
api integration - setup fee...$1200
tech support x12h @$95 = $1,140.00

sub: $2790
tax 7pct = 195.30
========
amt due $2,985.30

please remit by 04/02/2024
""",
            "ground_truth": {
                "invoice_number": "ts5892-b",
                "date": "2024-03-03",
                "vendor_name": "techstart solutions",
                "customer_name": "DataFlow",
                "subtotal": 2790.00,
                "tax": 195.30,
                "total": 2985.30,
                "line_items": [
                    {"description": "hosting (cloud, monthly plan)", "quantity": 1, "unit_price": 450.00, "amount": 450.00},
                    {"description": "api integration - setup fee", "quantity": 1, "unit_price": 1200.00, "amount": 1200.00},
                    {"description": "tech support", "quantity": 12, "unit_price": 95.00, "amount": 1140.00},
                ],
            },
        },
        {
            "id": "messy_003",
            "text": """GLOBAL SUPPLY
2500 industrial pkwy detroit MI

inv GS-0147rev
20-Feb-2024

Riverside Mfg / cleveland OH

stl bolts 100ct boxes -- 50 @ 12.50 ea ........... 625
cu wire 500' rolls -- 8 @ 85 .................... 680
safety goggles 10pk -- 20 @ 35 .................. 700
weld rods bundle -- 15 @ 22 ea .................. 330

s/t   2335.00
tax     163.45
-----
GRAND TOTAL  $2498.45

net45
""",
            "ground_truth": {
                "invoice_number": "GS-0147rev",
                "date": "2024-02-20",
                "vendor_name": "GLOBAL SUPPLY",
                "customer_name": "Riverside Mfg",
                "subtotal": 2335.00,
                "tax": 163.45,
                "total": 2498.45,
                "line_items": [
                    {"description": "stl bolts 100ct boxes", "quantity": 50, "unit_price": 12.50, "amount": 625.00},
                    {"description": "cu wire 500' rolls", "quantity": 8, "unit_price": 85.00, "amount": 680.00},
                    {"description": "safety goggles 10pk", "quantity": 20, "unit_price": 35.00, "amount": 700.00},
                    {"description": "weld rods bundle", "quantity": 15, "unit_price": 22.00, "amount": 330.00},
                ],
            },
        },
    ],

    # =========================================================================
    # MULTI-DOCUMENT — Multiple sections, cross-references, adjustments
    # =========================================================================
    "multi_document": [
        {
            "id": "multi_001",
            "text": """=== PURCHASE ORDER ===
PO Number: PO-2024-0055
Date: January 10, 2024
Vendor: Acme Corporation
Buyer: Widget Co.

Ordered Items:
- 10x Widget Type A @ $25.00 = $250.00
- 5x Widget Type B @ $40.00 = $200.00
- 8hrs Consulting @ $75.00/hr = $600.00

PO Total: $1,050.00 (before tax)

=== INVOICE ===
Invoice Number: INV-2024-001
Reference PO: PO-2024-0055
Date: January 15, 2024

From: Acme Corporation, 123 Business Ave, New York, NY 10001
To: Widget Co., 456 Commerce St, Chicago, IL 60601

Description                Qty    Unit Price    Amount
Widget Type A               10      $25.00     $250.00
Widget Type B                5      $40.00     $200.00
Consulting Service           8      $75.00     $600.00

Subtotal: $1,050.00
Tax (8%): $84.00
Invoice Total: $1,134.00

=== CREDIT MEMO ===
Credit Memo #: CM-2024-003
Reference Invoice: INV-2024-001
Date: January 22, 2024

Reason: 2x Widget Type A received defective
Credit Amount: $50.00

=== SUMMARY ===
Original Invoice: $1,134.00
Credit Applied: -$50.00
Adjusted Balance Due: $1,084.00
""",
            "ground_truth": {
                "invoice_number": "INV-2024-001",
                "date": "2024-01-15",
                "vendor_name": "Acme Corporation",
                "customer_name": "Widget Co.",
                "subtotal": 1050.00,
                "tax": 84.00,
                "total": 1134.00,
                "po_number": "PO-2024-0055",
                "adjustment_reason": "2x Widget Type A received defective",
                "adjusted_total": 1084.00,
                "line_items": [
                    {"description": "Widget Type A", "quantity": 10, "unit_price": 25.00, "amount": 250.00},
                    {"description": "Widget Type B", "quantity": 5, "unit_price": 40.00, "amount": 200.00},
                    {"description": "Consulting Service", "quantity": 8, "unit_price": 75.00, "amount": 600.00},
                ],
            },
        },
        {
            "id": "multi_002",
            "text": """--- PURCHASE ORDER ---
PO#: PO-DF-2024-112
Issued: 2024-02-28
Requested By: DataFlow Inc., Austin TX
Vendor: TechStart Solutions LLC

Items Requested:
1. Cloud Hosting (Monthly) - 1 unit - $450.00 - $450.00
2. API Integration - 1 unit - $1,200.00 - $1,200.00
3. Tech Support - 10 hours - $95.00/hr - $950.00
   NOTE: Hours are estimated, bill actuals

PO Authorized Amount: $2,600.00 (pre-tax)

--- INVOICE ---
Invoice: TS-5892
Date: March 3, 2024
PO Reference: PO-DF-2024-112

From: TechStart Solutions LLC, 890 Innovation Dr Suite 200, San Francisco CA 94105
To: DataFlow Inc., 321 Analytics Blvd, Austin TX 78701

Service                       Qty   Rate        Amount
Cloud Hosting (Monthly)         1   $450.00    $450.00
API Integration Setup           1   $1,200.00  $1,200.00
Technical Support (actual hrs) 12   $95.00     $1,140.00

NOTE: Support hours exceeded PO estimate (10hrs) by 2hrs.
Overage pre-approved by J. Smith on 03/01/2024.

Subtotal: $2,790.00
Tax (7%): $195.30
Total: $2,985.30

--- PAYMENT RECEIPT ---
Receipt #: RCP-2024-0891
Date: March 15, 2024
Payment Method: ACH Transfer
Reference: TS-5892

Amount Paid: $2,000.00
Outstanding Balance: $985.30
Due By: April 2, 2024
""",
            "ground_truth": {
                "invoice_number": "TS-5892",
                "date": "2024-03-03",
                "vendor_name": "TechStart Solutions LLC",
                "customer_name": "DataFlow Inc.",
                "subtotal": 2790.00,
                "tax": 195.30,
                "total": 2985.30,
                "po_number": "PO-DF-2024-112",
                "adjustment_reason": "Partial payment applied",
                "adjusted_total": 985.30,
                "line_items": [
                    {"description": "Cloud Hosting (Monthly)", "quantity": 1, "unit_price": 450.00, "amount": 450.00},
                    {"description": "API Integration Setup", "quantity": 1, "unit_price": 1200.00, "amount": 1200.00},
                    {"description": "Technical Support (actual hrs)", "quantity": 12, "unit_price": 95.00, "amount": 1140.00},
                ],
            },
        },
        {
            "id": "multi_003",
            "text": """==== PURCHASE ORDER ====
PO: PO-RM-2024-033
Date: Feb 15, 2024
Buyer: Riverside Manufacturing, 780 Factory Rd, Cleveland OH
Supplier: Global Supplies Inc.
Budget Approved: $2,800.00

Requested:
- Steel Bolts Box/100: 50 boxes @ $12.50
- Copper Wire 500ft: 10 rolls @ $85.00
- Safety Goggles Pack/10: 20 packs @ $35.00
- Welding Rods Bundle: 15 bundles @ $22.00

==== INVOICE ====
Invoice: GS-2024-0147
Date: February 20, 2024
PO Ref: PO-RM-2024-033

Billed By: Global Supplies Inc., 2500 Industrial Parkway, Detroit MI 48201
Billed To: Riverside Manufacturing, 780 Factory Road, Cleveland OH 44101

Item                       Qty   Unit$     Total
Steel Bolts (Box/100)       50   $12.50    $625.00
Copper Wire (500ft Roll)     8   $85.00    $680.00
Safety Goggles (Pack/10)    20   $35.00    $700.00
Welding Rods (Bundle)       15   $22.00    $330.00

IMPORTANT: Copper Wire qty reduced from PO (10 to 8).
2 rolls backordered, will ship separately.

Subtotal: $2,335.00
Tax (7%): $163.45
Total Due: $2,498.45

==== BACKORDER NOTICE ====
Backorder #: BO-2024-0089
Reference: GS-2024-0147 / PO-RM-2024-033
Item: Copper Wire (500ft Roll)
Qty Backordered: 2
Unit Price: $85.00
Backorder Amount: $170.00
Estimated Ship Date: March 10, 2024

Total with Backorder: $2,498.45 + $170.00 = $2,668.45
(Backorder will be invoiced separately upon shipment)
""",
            "ground_truth": {
                "invoice_number": "GS-2024-0147",
                "date": "2024-02-20",
                "vendor_name": "Global Supplies Inc.",
                "customer_name": "Riverside Manufacturing",
                "subtotal": 2335.00,
                "tax": 163.45,
                "total": 2498.45,
                "po_number": "PO-RM-2024-033",
                "adjustment_reason": "Copper Wire qty reduced from PO, 2 rolls backordered",
                "adjusted_total": 2668.45,
                "line_items": [
                    {"description": "Steel Bolts (Box/100)", "quantity": 50, "unit_price": 12.50, "amount": 625.00},
                    {"description": "Copper Wire (500ft Roll)", "quantity": 8, "unit_price": 85.00, "amount": 680.00},
                    {"description": "Safety Goggles (Pack/10)", "quantity": 20, "unit_price": 35.00, "amount": 700.00},
                    {"description": "Welding Rods (Bundle)", "quantity": 15, "unit_price": 22.00, "amount": 330.00},
                ],
            },
        },
    ],
}


# Required fields per task (defines what the agent must extract)
TASK_REQUIRED_FIELDS = {
    "simple_invoice": [
        "invoice_number", "date", "vendor_name", "customer_name",
        "subtotal", "tax", "total", "line_items",
    ],
    "messy_invoice": [
        "invoice_number", "date", "vendor_name", "customer_name",
        "subtotal", "tax", "total", "line_items",
    ],
    "multi_document": [
        "invoice_number", "date", "vendor_name", "customer_name",
        "subtotal", "tax", "total", "line_items",
        "po_number", "adjustment_reason", "adjusted_total",
    ],
}


def get_document(task_name: str, doc_index: int = 0) -> dict:
    """Get a document and its metadata for a given task.

    Args:
        task_name: One of 'simple_invoice', 'messy_invoice', 'multi_document'
        doc_index: Index into the document pool (will wrap around)

    Returns:
        dict with 'id', 'text', 'ground_truth', 'required_fields'
    """
    docs = DOCUMENTS.get(task_name, DOCUMENTS["simple_invoice"])
    doc = docs[doc_index % len(docs)]
    return {
        "id": doc["id"],
        "text": doc["text"],
        "ground_truth": doc["ground_truth"],
        "required_fields": TASK_REQUIRED_FIELDS.get(task_name, TASK_REQUIRED_FIELDS["simple_invoice"]),
    }
