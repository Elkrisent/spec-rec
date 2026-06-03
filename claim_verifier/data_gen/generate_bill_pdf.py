"""T7.1 — Generate a machine-readable text-layer hospital bill PDF."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_bill_pdf(facts: dict, output_path: str | Path) -> Path:
    """Create a hospital bill PDF with a machine-readable text layer.

    Required keys in facts:
        claim_id, hospital_name, hospital_address,
        admission_date (DD/MM/YYYY), discharge_date (DD/MM/YYYY),
        diagnosis, billed_amount (int), length_of_stay (int)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=A4)
    _, h = A4

    def row(y: float, label: str, value: str) -> None:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(60, y, f"{label} :")
        c.setFont("Helvetica", 10)
        c.drawString(230, y, value)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, h - 60, facts["hospital_name"].upper())
    c.setFont("Helvetica", 10)
    c.drawString(60, h - 78, facts.get("hospital_address", ""))

    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, h - 115, "HOSPITAL BILL / DISCHARGE SUMMARY")

    y = h - 150
    row(y,       "Claim ID",          facts["claim_id"])
    row(y - 18,  "Date of Admission", facts["admission_date"])
    row(y - 36,  "Date of Discharge", facts["discharge_date"])
    row(y - 54,  "Primary Diagnosis", facts["diagnosis"])
    row(y - 72,  "Length of Stay",    f"{facts['length_of_stay']} days")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y - 110, f"TOTAL BILL AMOUNT Rs.{facts['billed_amount']:,}")

    c.save()
    return path
