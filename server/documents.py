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

    # =========================================================================
    # CORRUPTED SCAN — OCR-like artifacts, character substitutions, garbled text
    # These simulate real scanned/faxed invoices with OCR errors.
    # =========================================================================
    "corrupted_scan": [
        {
            "id": "corrupt_001",
            "text": """SC4NNED D0CUMENT - Page 1 of 1

lNVOlCE

lnvoice Nurnber: lNV-2O24-OO1
Dat.e: Januery 1S, 2O24

Frorn:
  Acrne Corporati0n
  l23 Business Avenue
  New Y0rk, NY 1OOO1

BilI To:
  Widget C0.
  4S6 Cornmerce Street
  Chicag0, lL 6O6O1

Descripti0n                Qty    Unit Price    Arnount
---------------------------------------------------------
Widget Type A               1O      $2S.OO     $2SO.OO
Widget Type 8                S      $4O.OO     $2OO.OO
ConsuIting Service           8      $7S.OO     $6OO.OO

                                   Subtotal:  $1,OSO.OO
                                   Tax (8%):     $84.OO
                                   T0tal:     $1,l34.OO

Payrnent Terrns: Net 3O

--- END 0F SCAN ---
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
            "id": "corrupt_002",
            "text": """[SCAN QUALITY: P00R - SOME CHARACTERS MAY BE lNCORRECT]

TECHSTART S0LUTl0NS LLC
89O lnnovation Dr, Suite 2OO
San Francisc0, CA 941OS

lNV0lCE #: TS~S892
DATE: O3/O3/2O24

CUSTOMERr DataFIow lnc.
          321 AnaIytics BIvd
          Austin, TX 787O1

Servicc                       Qty   Unit Pricc     Total
----------------------------------------------------------
CIoud Hosting (MonthIy)         l     $4SO.OO    $4SO.OO
APl lntegration Setup           l   $l,2OO.OO  $l,2OO.OO
TechnicaI Support (hours)      l2      $9S.OO  $l,l4O.OO

                                    SubtotaI:  $2,79O.OO
                                    Tax (7%)):    $l9S.3O
                                    TotaI:     $2,98S.3O

Due Date: ApriI 2, 2O24

[PAGE 1/1 - SCAN C0MPLETE]
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
            "id": "corrupt_003",
            "text": """---FAXED DOCUMENT---
RECEIVED: 02/20/2024 14:32
QUALITY: [####===---] 40%

GL0BAL SUPPLlES lNC.
25OO lndustriaI Parkway
Detr0it, Ml 482Ol

lNVOlCE

lnvoice Number: GS-2O24-Ol47
Date: February 2O, 2024

T0:
  Riverside Manufactur1ng
  78O Factory R0ad
  CIeveIand, 0H 44l0l

Product                    Qty    Price Each    Line Total
-----------------------------------------------------------
SteeI BoIts (Box/lOO)       SO       $l2.SO       $62S.OO
Copper Wire (SOOft RoII)     8       $8S.OO       $68O.OO
Safety GoggIes (Pack/lO)    2O       $3S.OO       $7OO.OO
WeIding Rods (BundIe)       lS       $22.OO       $33O.OO

                   [iIIegibIe]
                                    SubtotaI:   $2,33S.OO
                                    SaIes Tax:    $l63.4S
                                    lnvoice T0tal: $2,498.4S

Terrns: Net 4S
---END FAX---
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
    # ADVERSARIAL INVOICE — Decoy fields, contradictions, hidden calculations
    # Designed to genuinely challenge frontier models with traps.
    # =========================================================================
    "adversarial_invoice": [
        {
            "id": "adversarial_001",
            "text": """INVOICE

*** IMPORTANT: This replaces previous invoice DRAFT-INV-999 which was voided ***

Invoice Number: INV-2024-001-R2
Previous Reference: DRAFT-INV-999 (VOIDED — DO NOT USE)
Date: January 15, 2024
Reissue Date: January 20, 2024

From:
  Acme Corporation
  123 Business Avenue, New York, NY 10001
  Tax ID: 12-3456789

Bill To:
  Widget Co. (DBA "WidgetCorp International")
  456 Commerce Street, Chicago, IL 60601
  Customer Account: WC-0042

Description                Qty    Unit Price    Amount
---------------------------------------------------------
Widget Type A               10      $25.00     $250.00
Widget Type B                5      $40.00     $200.00
Consulting Service           8      $75.00     $600.00
  ** EARLY PAYMENT DISCOUNT: -5% on consulting **

                                   Subtotal:  $1,050.00
                              Discount (5%):    -$30.00
                         Adjusted Subtotal:   $1,020.00
                                   Tax (8%):     $81.60
                                   Total:     $1,101.60

NOTE: Original quote (QT-2024-555) was $1,134.00 but discount applied.
Per agreement dated Jan 12, if paid within 10 days.

Payment Terms: Net 10 (discounted) / Net 30 (full price $1,134.00)
""",
            "ground_truth": {
                "invoice_number": "INV-2024-001-R2",
                "date": "2024-01-20",
                "vendor_name": "Acme Corporation",
                "customer_name": "Widget Co.",
                "subtotal": 1020.00,
                "tax": 81.60,
                "total": 1101.60,
                "discount_amount": 30.00,
                "original_total": 1134.00,
                "line_items": [
                    {"description": "Widget Type A", "quantity": 10, "unit_price": 25.00, "amount": 250.00},
                    {"description": "Widget Type B", "quantity": 5, "unit_price": 40.00, "amount": 200.00},
                    {"description": "Consulting Service", "quantity": 8, "unit_price": 75.00, "amount": 600.00},
                ],
                "discrepancy_notes": "5% early payment discount applied to consulting. Reissued invoice replaces voided DRAFT-INV-999. Adjusted subtotal $1,020 vs original $1,050.",
            },
        },
        {
            "id": "adversarial_002",
            "text": """--- PURCHASE ORDER ---
PO#: PO-DF-2024-112
Date: February 28, 2024
Vendor: TechStart Solutions LLC
Buyer: DataFlow Inc.
Authorized Budget: $2,600.00 (pre-tax)

Items:
1. Cloud Hosting - 1 unit @ $450.00 = $450.00
2. API Integration - 1 unit @ $1,200.00 = $1,200.00
3. Tech Support - 10 hours @ $95.00/hr = $950.00
PO Total: $2,600.00

--- INVOICE ---
Invoice: TS-5892-FINAL
Date: March 3, 2024
PO Reference: PO-DF-2024-112

From: TechStart Solutions LLC
To: DataFlow Inc.

Service                       Qty   Rate        Amount
Cloud Hosting (Monthly)         1   $450.00    $450.00
API Integration Setup           1   $1,200.00  $1,200.00
Technical Support (actual)     12   $95.00     $1,140.00
  >> 2 hrs over PO estimate, approved by J. Smith 03/01/2024
Rush Processing Fee             1   $150.00    $150.00
  >> Added per emergency request ER-2024-033

Subtotal: $2,940.00
Tax (7%): $205.80
Total: $3,145.80

!!! BUDGET VARIANCE ALERT !!!
PO Authorized: $2,600.00
Actual (pre-tax): $2,940.00
Variance: $340.00 OVER BUDGET (13.1%)
Causes: Support overage ($190), Rush fee ($150)

--- PAYMENT SCHEDULE ---
Payment 1 (due 03/15): $1,500.00
Payment 2 (due 04/02): $1,645.80
""",
            "ground_truth": {
                "invoice_number": "TS-5892-FINAL",
                "date": "2024-03-03",
                "vendor_name": "TechStart Solutions LLC",
                "customer_name": "DataFlow Inc.",
                "subtotal": 2940.00,
                "tax": 205.80,
                "total": 3145.80,
                "po_number": "PO-DF-2024-112",
                "discount_amount": 0.00,
                "original_total": 2600.00,
                "line_items": [
                    {"description": "Cloud Hosting (Monthly)", "quantity": 1, "unit_price": 450.00, "amount": 450.00},
                    {"description": "API Integration Setup", "quantity": 1, "unit_price": 1200.00, "amount": 1200.00},
                    {"description": "Technical Support (actual)", "quantity": 12, "unit_price": 95.00, "amount": 1140.00},
                    {"description": "Rush Processing Fee", "quantity": 1, "unit_price": 150.00, "amount": 150.00},
                ],
                "discrepancy_notes": "Invoice exceeds PO by $340 (13.1%). 2 extra support hours ($190) and rush processing fee ($150) added. PO authorized $2,600 but actual pre-tax is $2,940.",
            },
        },
        {
            "id": "adversarial_003",
            "text": """CONSOLIDATED STATEMENT

Account: Riverside Manufacturing
Statement Period: February 2024
Prepared by: Global Supplies Inc., Accounts Receivable

=== TRANSACTION 1: ORIGINAL INVOICE ===
Invoice: GS-2024-0147
Date: February 20, 2024
PO: PO-RM-2024-033

Steel Bolts (Box/100)       50   @ $12.50    =    $625.00
Copper Wire (500ft Roll)    10   @ $85.00    =    $850.00
Safety Goggles (Pack/10)    20   @ $35.00    =    $700.00
Welding Rods (Bundle)       15   @ $22.00    =    $330.00

Invoice Subtotal: $2,505.00
Tax (7%): $175.35
Invoice Total: $2,680.35

=== TRANSACTION 2: ADJUSTMENT ===
Credit Memo: CM-2024-0201
Date: February 25, 2024
Reference: GS-2024-0147

Issue: Copper Wire — only 8 of 10 rolls delivered.
2 rolls backordered (BO-2024-0089).
Credit for undelivered: 2 x $85.00 = $170.00
Tax adjustment: -$11.90
Total Credit: -$181.90

=== TRANSACTION 3: PRICE CORRECTION ===
Debit Memo: DM-2024-0055
Date: February 27, 2024

Steel Bolts price was quoted at $12.50 but contract
rate is $13.00. Underbilled on 50 boxes.
Price difference: 50 x $0.50 = $25.00
Tax on adjustment: $1.75
Total Debit: $26.75

=== ACCOUNT SUMMARY ===
Original Invoice:           $2,680.35
Credit (undelivered wire): -$181.90
Debit (price correction):   +$26.75
================================
Net Amount Due:             $2,525.20

Payment due by: March 20, 2024
""",
            "ground_truth": {
                "invoice_number": "GS-2024-0147",
                "date": "2024-02-20",
                "vendor_name": "Global Supplies Inc.",
                "customer_name": "Riverside Manufacturing",
                "subtotal": 2505.00,
                "tax": 175.35,
                "total": 2680.35,
                "po_number": "PO-RM-2024-033",
                "discount_amount": 0.00,
                "original_total": 2680.35,
                "line_items": [
                    {"description": "Steel Bolts (Box/100)", "quantity": 50, "unit_price": 12.50, "amount": 625.00},
                    {"description": "Copper Wire (500ft Roll)", "quantity": 10, "unit_price": 85.00, "amount": 850.00},
                    {"description": "Safety Goggles (Pack/10)", "quantity": 20, "unit_price": 35.00, "amount": 700.00},
                    {"description": "Welding Rods (Bundle)", "quantity": 15, "unit_price": 22.00, "amount": 330.00},
                ],
                "discrepancy_notes": "Credit memo CM-2024-0201 for 2 undelivered Copper Wire rolls (-$181.90). Debit memo DM-2024-0055 for Steel Bolts price correction (+$26.75). Net adjustment: -$155.15. Final amount due: $2,525.20.",
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
    "corrupted_scan": [
        "invoice_number", "date", "vendor_name", "customer_name",
        "subtotal", "tax", "total", "line_items",
    ],
    "adversarial_invoice": [
        "invoice_number", "date", "vendor_name", "customer_name",
        "subtotal", "tax", "total", "line_items",
        "po_number", "discount_amount", "original_total",
        "discrepancy_notes",
    ],
}


def get_document(task_name: str, doc_index: int = 0, use_procedural: bool = True) -> dict:
    """Get a document and its metadata for a given task.

    For doc_index 0-2, returns static documents (deterministic test fixtures).
    For doc_index >= 3 (or when use_procedural=True and index wraps), uses the
    procedural generation engine to create novel documents from the seed.

    Args:
        task_name: One of 'simple_invoice', 'messy_invoice', 'multi_document',
                   'corrupted_scan', 'adversarial_invoice'
        doc_index: Index / seed for document selection
        use_procedural: Whether to use procedural generation for indices beyond static pool

    Returns:
        dict with 'id', 'text', 'ground_truth', 'required_fields'
    """
    docs = DOCUMENTS.get(task_name, DOCUMENTS["simple_invoice"])
    required = TASK_REQUIRED_FIELDS.get(task_name, TASK_REQUIRED_FIELDS["simple_invoice"])

    # Use static documents for small indices (deterministic test fixtures)
    if doc_index < len(docs):
        doc = docs[doc_index]
        return {
            "id": doc["id"],
            "text": doc["text"],
            "ground_truth": doc["ground_truth"],
            "required_fields": required,
        }

    # Use procedural generation for larger indices
    if use_procedural:
        from .procedural import generate_document
        proc_doc = generate_document(task_name, seed=doc_index)
        return {
            "id": proc_doc["id"],
            "text": proc_doc["text"],
            "ground_truth": proc_doc["ground_truth"],
            "required_fields": required,
        }

    # Fallback: wrap around static docs
    doc = docs[doc_index % len(docs)]
    return {
        "id": doc["id"],
        "text": doc["text"],
        "ground_truth": doc["ground_truth"],
        "required_fields": required,
    }

