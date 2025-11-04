"""
Portfolio PDF Exporter Module.

This module provides functionality to export portfolio analysis
and recommendations as a professional PDF document.
"""

import io
from datetime import datetime
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PortfolioPDFExporter:
    """
    Exporter class for generating PDF reports of portfolio analysis.
    """

    def __init__(self):
        """Initialize PDF exporter with styles."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF."""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=30,
                alignment=1,  # Center alignment
            )
        )

        # Subtitle style
        self.styles.add(
            ParagraphStyle(
                name="CustomSubtitle",
                parent=self.styles["Heading1"],
                fontSize=16,
                textColor=colors.HexColor("#2563eb"),
                spaceAfter=12,
            )
        )

        # Section header style
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=10,
                spaceBefore=15,
            )
        )

        # Body text style
        self.styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=self.styles["Normal"],
                fontSize=10,
                spaceAfter=8,
                leading=14,
            )
        )

        # Small text style
        self.styles.add(
            ParagraphStyle(
                name="SmallText",
                parent=self.styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
                spaceAfter=6,
            )
        )

    def generate_pdf(
        self,
        portfolio: Dict,
        financial_profile: Optional[Dict] = None,
        provider: str = "AI",
        model: str = "Unknown",
    ) -> bytes:
        """
        Generate a PDF document containing portfolio analysis.

        Args:
            portfolio: Portfolio dictionary with assets and recommendations
            financial_profile: Optional financial profile dictionary
            provider: LLM provider name
            model: Model name used for generation

        Returns:
            PDF document as bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Build content
        story = []

        # Add cover page
        story.extend(self._create_cover_page(provider, model))
        story.append(PageBreak())

        # Add financial profile summary if available
        if financial_profile:
            story.extend(self._create_profile_section(financial_profile))
            story.append(PageBreak())

        # Add portfolio recommendation
        story.extend(self._create_portfolio_section(portfolio))

        # Add risk analysis
        story.extend(self._create_risk_section(portfolio))

        # Add disclaimers
        story.append(PageBreak())
        story.extend(self._create_disclaimer_section())

        # Build PDF
        doc.build(story)

        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def _create_cover_page(self, provider: str, model: str) -> list:
        """Create the cover page content."""
        content = []

        # Add spacer for vertical centering
        content.append(Spacer(1, 2 * inch))

        # Title
        content.append(
            Paragraph(
                "Portfolio Analysis Report",
                self.styles["CustomTitle"],
            )
        )

        content.append(Spacer(1, 0.5 * inch))

        # Generation info
        generation_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        content.append(
            Paragraph(
                f"<b>Generated:</b> {generation_date}",
                self.styles["CustomBody"],
            )
        )

        content.append(
            Paragraph(
                f"<b>AI Provider:</b> {provider}",
                self.styles["CustomBody"],
            )
        )

        content.append(
            Paragraph(
                f"<b>Model:</b> {model}",
                self.styles["CustomBody"],
            )
        )

        content.append(Spacer(1, 1 * inch))

        # Privacy notice
        content.append(
            Paragraph(
                "<b>⚠️ Confidential Document</b>",
                self.styles["SectionHeader"],
            )
        )
        content.append(
            Paragraph(
                "This document contains sensitive financial information. "
                "Please store it securely and share only with trusted advisors.",
                self.styles["SmallText"],
            )
        )

        return content

    def _create_profile_section(self, profile: Dict) -> list:
        """Create financial profile summary section."""
        content = []

        content.append(
            Paragraph(
                "Financial Profile Summary",
                self.styles["CustomSubtitle"],
            )
        )

        content.append(Spacer(1, 0.2 * inch))

        # Create profile table
        profile_data = []

        # Define fields to display
        fields = [
            ("Age Range", "age_range"),
            ("Employment Status", "employment_status"),
            ("Annual Income Range", "annual_income_range"),
            ("Monthly Savings", "monthly_savings_amount"),
            ("Investment Experience", "investment_experience"),
            ("Risk Tolerance", "risk_tolerance"),
            ("Financial Goals", "goals"),
            ("Geographic Allocation", "geographic_allocation"),
        ]

        for label, key in fields:
            value = profile.get(key, "N/A")
            if value and value not in ["None", "N/A", "", None]:
                profile_data.append([label, str(value)])

        if profile_data:
            table = Table(profile_data, colWidths=[2.5 * inch, 3.5 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            content.append(table)

        content.append(Spacer(1, 0.3 * inch))

        # Add summary notes if available
        if profile.get("summary_notes") and profile["summary_notes"] not in [
            "None",
            "N/A",
        ]:
            content.append(
                Paragraph(
                    "<b>Summary Notes:</b>",
                    self.styles["SectionHeader"],
                )
            )
            content.append(
                Paragraph(
                    profile["summary_notes"],
                    self.styles["CustomBody"],
                )
            )

        return content

    def _get_asset_field(self, asset, field: str):
        """
        Helper method to safely extract asset field from dict or object.
        
        Args:
            asset: Asset object or dictionary
            field: Field name to extract
            
        Returns:
            Field value or None
        """
        if isinstance(asset, dict):
            return asset.get(field)
        return getattr(asset, field, None)

    def _create_portfolio_section(self, portfolio: Dict) -> list:
        """Create portfolio recommendation section."""
        content = []

        content.append(
            Paragraph(
                "Portfolio Recommendation",
                self.styles["CustomSubtitle"],
            )
        )

        content.append(Spacer(1, 0.2 * inch))

        # Overall strategy
        if "portfolio_reasoning" in portfolio:
            content.append(
                Paragraph(
                    "<b>Investment Strategy:</b>",
                    self.styles["SectionHeader"],
                )
            )
            content.append(
                Paragraph(
                    portfolio["portfolio_reasoning"],
                    self.styles["CustomBody"],
                )
            )
            content.append(Spacer(1, 0.2 * inch))

        # Asset allocation table
        content.append(
            Paragraph(
                "<b>Asset Allocation:</b>",
                self.styles["SectionHeader"],
            )
        )

        if "assets" in portfolio and isinstance(portfolio["assets"], list):
            # Create asset allocation table
            asset_data = [["Asset", "Allocation %", "Justification"]]

            for asset in portfolio["assets"]:
                symbol = self._get_asset_field(asset, "symbol")
                percentage = self._get_asset_field(asset, "percentage")
                justification = self._get_asset_field(asset, "justification")

                # Truncate justification if too long
                if justification and len(justification) > 120:
                    justification = justification[:117] + "..."

                asset_data.append([symbol, f"{percentage}%", justification or ""])

            table = Table(asset_data, colWidths=[1.2 * inch, 1 * inch, 3.5 * inch])
            table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#1f4788"),
                        ),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("ALIGN", (1, 0), (1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            content.append(table)

        content.append(Spacer(1, 0.3 * inch))

        # Key considerations
        if "key_considerations" in portfolio and portfolio["key_considerations"]:
            content.append(
                Paragraph(
                    "<b>Key Considerations:</b>",
                    self.styles["SectionHeader"],
                )
            )

            considerations = portfolio["key_considerations"]
            if isinstance(considerations, list):
                for consideration in considerations:
                    if consideration:
                        content.append(
                            Paragraph(
                                f"• {consideration}",
                                self.styles["CustomBody"],
                            )
                        )
            else:
                # Handle string format
                for consideration in str(considerations).split(";"):
                    consideration = consideration.strip()
                    if consideration:
                        content.append(
                            Paragraph(
                                f"• {consideration}",
                                self.styles["CustomBody"],
                            )
                        )

        content.append(Spacer(1, 0.2 * inch))

        # Rebalancing schedule
        if "rebalancing_schedule" in portfolio:
            content.append(
                Paragraph(
                    f"<b>Rebalancing Schedule:</b> {portfolio['rebalancing_schedule']}",
                    self.styles["CustomBody"],
                )
            )

        return content

    def _create_risk_section(self, portfolio: Dict) -> list:
        """Create risk analysis section."""
        content = []

        content.append(Spacer(1, 0.3 * inch))

        content.append(
            Paragraph(
                "Risk Analysis",
                self.styles["CustomSubtitle"],
            )
        )

        content.append(Spacer(1, 0.2 * inch))

        # Risk level
        if "risk_level" in portfolio:
            risk_value = portfolio["risk_level"]
            if isinstance(risk_value, str):
                risk_level = risk_value.upper()
            else:
                risk_level = str(risk_value).replace("RiskLevel.", "").upper()

            content.append(
                Paragraph(
                    f"<b>Risk Level:</b> {risk_level}",
                    self.styles["SectionHeader"],
                )
            )

            # Add risk description based on level
            risk_descriptions = {
                "CONSERVATIVE": "This portfolio prioritizes capital preservation and stable returns. "
                "It typically includes a higher allocation to bonds and cash equivalents, "
                "with lower volatility but potentially lower long-term returns.",
                "MODERATE": "This balanced portfolio seeks a middle ground between growth and stability. "
                "It includes a mix of stocks and bonds designed to provide reasonable returns "
                "while managing risk through diversification.",
                "AGGRESSIVE": "This growth-oriented portfolio aims for higher long-term returns "
                "through increased equity exposure. It accepts higher volatility and short-term "
                "fluctuations in pursuit of greater capital appreciation.",
            }

            if risk_level in risk_descriptions:
                content.append(
                    Paragraph(
                        risk_descriptions[risk_level],
                        self.styles["CustomBody"],
                    )
                )

        content.append(Spacer(1, 0.2 * inch))

        # General risk warnings
        content.append(
            Paragraph(
                "<b>Volatility Expectations:</b>",
                self.styles["SectionHeader"],
            )
        )
        content.append(
            Paragraph(
                "All investments carry risk, including the potential loss of principal. "
                "Market volatility can cause short-term fluctuations in portfolio value. "
                "The projected returns and risk levels are based on historical data and "
                "may not reflect future performance.",
                self.styles["CustomBody"],
            )
        )

        return content

    def _create_disclaimer_section(self) -> list:
        """Create disclaimer section."""
        content = []

        content.append(
            Paragraph(
                "Important Disclaimers",
                self.styles["CustomSubtitle"],
            )
        )

        content.append(Spacer(1, 0.2 * inch))

        disclaimers = [
            {
                "title": "Not Financial Advice",
                "text": "This report is generated by an AI system and is provided for "
                "informational purposes only. It does not constitute professional financial "
                "advice, investment recommendations, or a solicitation to buy or sell securities. "
                "You should consult with a qualified financial advisor before making any "
                "investment decisions.",
            },
            {
                "title": "Past Performance",
                "text": "Past performance is not indicative of future results. Historical returns "
                "and projections shown in this report are based on historical data and models "
                "that may not accurately predict future market conditions or investment outcomes.",
            },
            {
                "title": "Risk Disclosure",
                "text": "All investments involve risk, including the potential loss of principal. "
                "There is no guarantee that any investment strategy will be successful or that "
                "investment objectives will be achieved. Different types of investments involve "
                "varying degrees of risk, and there can be no assurance that any specific "
                "investment will be suitable or profitable for your situation.",
            },
            {
                "title": "No Guarantees",
                "text": "The recommendations and projections in this report are based on current "
                "market conditions, assumptions, and the information provided. Market conditions "
                "can change rapidly, and actual results may differ materially from projections. "
                "Neither the AI system nor its creators guarantee any specific investment results.",
            },
            {
                "title": "Individual Circumstances",
                "text": "This report is generated based on general information and may not fully "
                "account for your unique financial situation, tax considerations, or investment "
                "objectives. Professional financial advice should consider your complete financial "
                "picture, including factors not captured in this analysis.",
            },
            {
                "title": "Accuracy of Information",
                "text": "While efforts are made to ensure accuracy, the information in this report "
                "may contain errors or omissions. Financial data, market information, and asset "
                "details should be independently verified before making investment decisions.",
            },
        ]

        for disclaimer in disclaimers:
            content.append(
                Paragraph(
                    f"<b>{disclaimer['title']}:</b>",
                    self.styles["SectionHeader"],
                )
            )
            content.append(
                Paragraph(
                    disclaimer["text"],
                    self.styles["SmallText"],
                )
            )
            content.append(Spacer(1, 0.15 * inch))

        content.append(Spacer(1, 0.2 * inch))

        # Footer
        generation_info = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        content.append(
            Paragraph(
                f"<i>{generation_info}</i>",
                self.styles["SmallText"],
            )
        )

        return content

    def generate_filename(self, prefix: str = "portfolio_analysis") -> str:
        """
        Generate a filename for the PDF with timestamp.

        Args:
            prefix: Prefix for the filename

        Returns:
            Filename string with timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{prefix}_{timestamp}.pdf"
