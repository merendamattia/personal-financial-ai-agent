Extract the PAC (Piano di Accumulo del Capitale) metrics from this financial profile.

Financial Profile:
{financial_profile}

Extract ONLY these 2 values:
1. **initial_investment**: How much money the user has saved or can invest as a lump sum (in euros).
   - Look at 'investments' field
   - Convert ranges to numeric values: "1k-5k" → 3000, "5k-20k" → 12500, "20k+" → 30000

2. **monthly_savings**: How much the user can save and invest monthly (in euros).
   - Look at 'savings_amount' field
   - Minimum 100 euros

Return ONLY these 2 numeric values in the exact JSON structure required.
