Extract ONLY the following 17 financial profile fields from the conversation summary.

CRITICAL: Return ONLY these exact field names. Do NOT create any additional fields.

The 17 fields are:
1. age_range - Age range (e.g., '25-34', '35-44', etc.)
2. employment_status - Employment status (e.g., 'employed', 'self-employed', 'retired')
3. annual_income_range - Income range (e.g., '30k-50k', '50k-100k', '100k+')
4. income_stability - Income stability (e.g., 'stable', 'moderate', 'unstable')
5. monthly_expenses_range - Monthly expenses range (e.g., '2k-3k', '3k-5k')
6. major_expenses - Major recurring expenses (mortgage, rent, car payment)
7. total_debt - Total outstanding debt (e.g., 'minimal', '10k-50k', '50k-100k')
8. debt_types - Types of debt (mortgage, credit card, student loans)
9. monthly_savings_amount - Amount in savings (e.g., '0', '100', '250', '1k+')
10. active_investments - Amount of active investments (stocks, ETFs, crypto)
11. investment_experience - Investment experience (beginner, intermediate, advanced)
12. goals - Primary financial goals
13. risk_tolerance - Risk tolerance (conservative, moderate, aggressive)
14. geographic_allocation - Geographic investment preference
15. family_dependents - Number of dependents or family situation
16. insurance_coverage - Types of insurance coverage
17. summary_notes - Any additional important notes

For any field not mentioned in the conversation, use the default values provided by the schema.
If information is not explicitly mentioned, do NOT fabricate values.

Conversation Summary:
{conversation_summary}

Return ONLY these 17 fields exactly as named above. Do not create extra fields.
