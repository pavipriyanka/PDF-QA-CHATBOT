"""Create data/sample.pdf, a small 3-page handbook used by the demo, tests and evaluation.

Usage:  python scripts/make_sample_pdf.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Acme Robotics Employee Handbook: Leave Policy", [
        "Full-time employees receive 24 days of paid annual leave per calendar year. "
        "Leave accrues at 2 days per month and up to 10 unused days can be carried forward to the next year.",
        "Sick leave is separate from annual leave. Employees get 12 days of paid sick leave per year. "
        "A medical certificate is required for sick leave longer than 3 consecutive days.",
        "Parental leave is 26 weeks for the primary caregiver and 4 weeks for the secondary caregiver.",
    ]),
    ("Remote Work and Equipment", [
        "Employees may work remotely up to 3 days per week with manager approval. "
        "Core collaboration hours are 11:00 to 16:00 IST, when everyone should be reachable.",
        "New joiners receive a laptop and a one-time home office allowance of INR 15,000 "
        "for a chair, desk or monitor. Receipts must be submitted within 30 days.",
        "Lost or damaged equipment must be reported to the IT helpdesk using ticket code IT-HW-07.",
    ]),
    ("Performance Reviews and Learning", [
        "Performance reviews happen twice a year, in April and October. "
        "Ratings use a 5-point scale and are calibrated across teams.",
        "Every employee has an annual learning budget of INR 40,000 for courses, books and conferences. "
        "Certifications relevant to the role are fully reimbursed after passing.",
        "The probation period for new employees is 6 months, after which confirmation is decided by the manager.",
    ]),
]


def main(out: Path = Path(__file__).resolve().parent.parent / "data" / "sample.pdf") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for title, paragraphs in PAGES:
        story += [Paragraph(title, styles["Heading1"]), Spacer(1, 12)]
        for p in paragraphs:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 10)]
        story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4).build(story[:-1])
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
