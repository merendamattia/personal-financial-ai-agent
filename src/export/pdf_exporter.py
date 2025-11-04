"""
Portfolio PDF Exporter Module.

This module provides functionality to export portfolio analysis
and recommendations as a professional PDF document.
"""

import io
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
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

    def _create_pie_chart(self, portfolio: Dict) -> Optional[RLImage]:
        """
        Create a pie chart for portfolio asset allocation.

        Args:
            portfolio: Portfolio dictionary containing assets

        Returns:
            ReportLab Image object or None if chart cannot be created
        """
        try:
            if "assets" not in portfolio or not isinstance(portfolio["assets"], list):
                return None

            # Prepare data for pie chart
            asset_symbols = []
            asset_percentages = []

            for asset in portfolio["assets"]:
                symbol = self._get_asset_field(asset, "symbol")
                percentage = self._get_asset_field(asset, "percentage")
                if symbol and percentage:
                    asset_symbols.append(symbol)
                    asset_percentages.append(percentage)

            if not asset_symbols:
                return None

            # Create DataFrame for plotly
            df = pd.DataFrame({
                'Asset': asset_symbols,
                'Allocation': asset_percentages
            })

            # Create pie chart
            fig = px.pie(
                df,
                values='Allocation',
                names='Asset',
                hole=0.3,
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )

            fig.update_layout(
                height=220,
                width=450,
                showlegend=True,
                margin=dict(l=20, r=20, t=25, b=20),
                font=dict(size=8),
            )

            # Convert to image (requires kaleido)
            try:
                img_bytes = fig.to_image(format="png", width=450, height=220)
                img_buffer = io.BytesIO(img_bytes)

                # Create ReportLab Image
                rl_image = RLImage(img_buffer, width=4.5 * inch, height=2.2 * inch)
                return rl_image
            except Exception:
                # If kaleido is not available, return None gracefully
                return None

        except Exception:
            # If chart generation fails, return None
            # This ensures PDF generation continues even without charts
            return None

    def _create_compact_summary_page(
        self, portfolio: Dict, financial_profile: Optional[Dict], 
        provider: str, model: str
    ) -> list:
        """Create a compact summary page with all key information."""
        content = []
        
        # Header
        content.append(Paragraph(
            f"<b>Portfolio Analysis Report</b><br/>"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
            f"Provider: {provider} | Model: {model}",
            self.styles["CustomBody"]
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Financial Profile (compact)
        if financial_profile:
            content.append(Paragraph("<b>Financial Profile</b>", self.styles["SectionHeader"]))
            
            # Compact profile table
            profile_data = []
            key_fields = [
                ("Age", "age_range"),
                ("Risk", "risk_tolerance"),
                ("Experience", "investment_experience"),
                ("Goals", "goals"),
            ]
            
            for label, key in key_fields:
                value = financial_profile.get(key, "N/A")
                if value and value not in ["None", "N/A", ""]:
                    profile_data.append([label, str(value)[:50]])
            
            if profile_data:
                table = Table(profile_data, colWidths=[1 * inch, 4.5 * inch])
                table.setStyle(TableStyle([
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ]))
                content.append(table)
            
            content.append(Spacer(1, 0.15 * inch))
        
        # Portfolio Allocation (compact)
        content.append(Paragraph("<b>Portfolio Allocation</b>", self.styles["SectionHeader"]))
        
        if "assets" in portfolio and isinstance(portfolio["assets"], list):
            asset_data = [["Asset", "%", "Justification"]]
            
            for asset in portfolio["assets"]:
                symbol = self._get_asset_field(asset, "symbol")
                percentage = self._get_asset_field(asset, "percentage")
                justification = self._get_asset_field(asset, "justification")
                
                if justification and len(justification) > 80:
                    justification = justification[:77] + "..."
                
                asset_data.append([symbol, f"{percentage}%", justification or ""])
            
            table = Table(asset_data, colWidths=[0.8 * inch, 0.5 * inch, 4.2 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ]))
            content.append(table)
        
        content.append(Spacer(1, 0.15 * inch))
        
        # Strategy & Risk (compact)
        if "portfolio_reasoning" in portfolio:
            content.append(Paragraph("<b>Strategy</b>", self.styles["SectionHeader"]))
            # Truncate reasoning if too long
            reasoning = portfolio["portfolio_reasoning"]
            if len(reasoning) > 200:
                reasoning = reasoning[:197] + "..."
            content.append(Paragraph(reasoning, self.styles["SmallText"]))
            content.append(Spacer(1, 0.1 * inch))
        
        if "risk_level" in portfolio:
            risk_value = portfolio["risk_level"]
            if isinstance(risk_value, str):
                risk_level = risk_value.upper()
            else:
                risk_level = str(risk_value).replace("RiskLevel.", "").upper()
            
            content.append(Paragraph(
                f"<b>Risk Level:</b> {risk_level}", 
                self.styles["CustomBody"]
            ))
        
        # Key considerations (compact)
        if "key_considerations" in portfolio and portfolio["key_considerations"]:
            content.append(Spacer(1, 0.1 * inch))
            content.append(Paragraph("<b>Key Points</b>", self.styles["SectionHeader"]))
            considerations = portfolio["key_considerations"]
            if isinstance(considerations, list):
                for i, consideration in enumerate(considerations[:3]):  # Max 3
                    if consideration:
                        content.append(Paragraph(
                            f"• {consideration[:80]}{'...' if len(consideration) > 80 else ''}", 
                            self.styles["SmallText"]
                        ))
        
        # Disclaimer (compact)
        content.append(Spacer(1, 0.15 * inch))
        content.append(Paragraph(
            "<b>⚠ Disclaimer:</b> This report is AI-generated for informational purposes only. "
            "Not financial advice. Consult a qualified advisor before making investment decisions.",
            self.styles["SmallText"]
        ))
        
        return content
    
    def _create_charts_page(
        self, portfolio: Dict, 
        monte_carlo_lump_data: Optional[Dict],
        monte_carlo_pac_data: Optional[Dict]
    ) -> list:
        """Create a page with all visualization charts."""
        content = []
        
        content.append(Paragraph("<b>Portfolio Visualizations</b>", self.styles["CustomSubtitle"]))
        content.append(Spacer(1, 0.15 * inch))
        
        # Pie chart
        pie_chart = self._create_pie_chart(portfolio)
        if pie_chart:
            content.append(Paragraph("<b>Asset Allocation</b>", self.styles["SectionHeader"]))
            content.append(Spacer(1, 0.05 * inch))
            content.append(pie_chart)
            content.append(Spacer(1, 0.15 * inch))
        
        # Monte Carlo Lump Sum chart
        if monte_carlo_lump_data:
            mc_lump_chart = self._create_monte_carlo_chart(monte_carlo_lump_data)
            if mc_lump_chart:
                content.append(Paragraph("<b>Lump Sum Projection</b>", self.styles["SectionHeader"]))
                content.append(Spacer(1, 0.05 * inch))
                content.append(mc_lump_chart)
                content.append(Spacer(1, 0.15 * inch))
        
        # Monte Carlo PAC chart
        if monte_carlo_pac_data:
            mc_pac_chart = self._create_monte_carlo_chart(monte_carlo_pac_data)
            if mc_pac_chart:
                content.append(Paragraph("<b>PAC Projection</b>", self.styles["SectionHeader"]))
                content.append(Spacer(1, 0.05 * inch))
                content.append(mc_pac_chart)
        
        return content

    def _create_monte_carlo_chart(self, fig_data: Dict) -> Optional[RLImage]:
        """
        Create Monte Carlo simulation chart image.

        Args:
            fig_data: Dictionary containing plotly figure data
                     (x, y arrays for different percentiles)

        Returns:
            ReportLab Image object or None if chart cannot be created
        """
        try:
            if not fig_data or 'x' not in fig_data:
                return None

            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add traces if they exist in fig_data
            if 'percentile_10' in fig_data:
                fig.add_trace(go.Scatter(
                    x=fig_data['x'],
                    y=fig_data['percentile_10'],
                    name="Pessimistic (10th)",
                    line=dict(color="rgba(255, 0, 0, 0.3)", width=2),
                ))
            
            if 'percentile_75' in fig_data:
                fig.add_trace(go.Scatter(
                    x=fig_data['x'],
                    y=fig_data['percentile_75'],
                    name="Optimistic (75th)",
                    line=dict(color="rgba(0, 255, 0, 0.3)", width=2),
                    fill='tonexty',
                ))
            
            if 'percentile_50' in fig_data:
                fig.add_trace(go.Scatter(
                    x=fig_data['x'],
                    y=fig_data['percentile_50'],
                    name="Expected (Median)",
                    line=dict(color="rgb(0, 100, 200)", width=3),
                ))
            
            # Add cumulative invested for PAC charts
            if 'cumulative_invested' in fig_data:
                fig.add_trace(go.Scatter(
                    x=fig_data['x'],
                    y=fig_data['cumulative_invested'],
                    name="Total Invested",
                    line=dict(color="rgb(50, 50, 150)", width=2, dash="dash"),
                ))
            
            fig.update_layout(
                title=fig_data.get('title', 'Wealth Projection'),
                xaxis_title="Years",
                yaxis_title="Value (€)",
                height=220,
                width=450,
                font=dict(size=8),
                margin=dict(l=40, r=40, t=35, b=35),
                showlegend=True,
                legend=dict(font=dict(size=7))
            )
            
            try:
                img_bytes = fig.to_image(format="png", width=450, height=220)
                img_buffer = io.BytesIO(img_bytes)
                rl_image = RLImage(img_buffer, width=4.5 * inch, height=2.2 * inch)
                return rl_image
            except Exception:
                return None
                
        except Exception:
            return None

    def generate_pdf(
        self,
        portfolio: Dict,
        financial_profile: Optional[Dict] = None,
        provider: str = "AI",
        model: str = "Unknown",
        include_charts: bool = True,
        monte_carlo_lump_data: Optional[Dict] = None,
        monte_carlo_pac_data: Optional[Dict] = None,
    ) -> bytes:
        """
        Generate a compact 2-page PDF document containing portfolio analysis.

        Args:
            portfolio: Portfolio dictionary with assets and recommendations
            financial_profile: Optional financial profile dictionary
            provider: LLM provider name
            model: Model name used for generation
            include_charts: Whether to include charts/graphs in the PDF (default: True)
            monte_carlo_lump_data: Optional Monte Carlo lump sum simulation data
            monte_carlo_pac_data: Optional Monte Carlo PAC simulation data

        Returns:
            PDF document as bytes (2 pages)
        """
        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
        )

        # Build content
        story = []

        # PAGE 1: Summary information
        story.extend(self._create_compact_summary_page(
            portfolio, financial_profile, provider, model
        ))
        
        # PAGE 2: All graphs
        if include_charts:
            story.append(PageBreak())
            story.extend(self._create_charts_page(
                portfolio, monte_carlo_lump_data, monte_carlo_pac_data
            ))

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

    def _create_portfolio_section(self, portfolio: Dict, include_charts: bool = True) -> list:
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

        content.append(Spacer(1, 0.2 * inch))

        # Add pie chart if charts are enabled
        if include_charts:
            pie_chart = self._create_pie_chart(portfolio)
            if pie_chart:
                content.append(
                    Paragraph(
                        "<b>Asset Allocation Visualization:</b>",
                        self.styles["SectionHeader"],
                    )
                )
                content.append(Spacer(1, 0.1 * inch))
                content.append(pie_chart)
                content.append(Spacer(1, 0.1 * inch))

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
