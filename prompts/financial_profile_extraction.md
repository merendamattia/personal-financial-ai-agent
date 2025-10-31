Extract ONLY the following 19 financial profile fields from the conversation summary.

CRITICAL: Return ONLY these exact field names. Do NOT create any additional fields.

The 19 fields are:
1. age_range - Age range (e.g., '25-34', '35-44', etc.)
2. employment_status - Employment status (e.g., 'employed', 'self-employed', 'retired')
3. annual_income_range - Income range (e.g., '30k-50k', '50k-100k', '100k+')
4. income_stability - Income stability (e.g., 'stable', 'moderate', 'unstable')
5. additional_income_sources - Additional income sources
6. monthly_expenses_range - Monthly expenses range (e.g., '2k-3k', '3k-5k')
7. major_expenses - Major recurring expenses (mortgage, rent, car payment)
8. total_debt - Total outstanding debt (e.g., 'minimal', '10k-50k', '50k-100k')
9. debt_types - Types of debt (mortgage, credit card, student loans)
10. savings_amount - Amount in savings (e.g., 'none', '1k-5k', '5k-20k', '20k+')
11. investments - Investment portfolio details (stocks, ETFs, crypto)
12. investment_experience - Investment experience (beginner, intermediate, advanced)
13. goals - Primary financial goals
14. risk_tolerance - Risk tolerance (conservative, moderate, aggressive)
15. risk_concerns - Specific financial concerns or risks
16. geographic_allocation - Geographic investment preference
17. family_dependents - Number of dependents or family situation
18. insurance_coverage - Types of insurance coverage
19. summary_notes - Any additional important notes

For any field not mentioned in the conversation, use the default values provided by the schema.
If information is not explicitly mentioned, do NOT fabricate values.

Conversation Summary:
{conversation_summary}

Return ONLY these 19 fields exactly as named above. Do not create extra fields.
