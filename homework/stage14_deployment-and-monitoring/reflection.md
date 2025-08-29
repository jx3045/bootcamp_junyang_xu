# Stage 14: Deployment & Monitoring — Reflection

Our current project model, trained on processed financial transaction data (including features such as region, age, income, and transaction counts), would face several risks if deployed into production. On the **data layer**, schema drift could occur if upstream pipelines change column names or types. Missing values may increase unexpectedly, especially for income or spend fields, which would degrade predictions. Delayed data ingestion could also reduce freshness and cause business reports to lag.  

On the **model layer**, the regression model may suffer from concept drift if user behavior or income distributions shift. For example, a 5% increase in Population Stability Index (PSI) on key features like income should trigger retraining. Rolling Mean Absolute Error (MAE) is another critical metric; if 7-day MAE rises 10% above baseline, we would investigate calibration and retrain as needed.  

On the **system layer**, we must ensure API reliability and responsiveness. A p95 latency above 250ms or an error rate greater than 2% should immediately notify the platform on-call team. Batch job success rate should remain above 98%.  

On the **business layer**, monitoring KPIs such as approval rate (target > 60%) and bad rate (target < 10%) ensures alignment with financial outcomes. If these deviate, analysts should review and escalate.  

**Ownership** is divided: Data Engineering handles schema and null checks, ML Engineering monitors MAE and drift, Platform Ops ensures system SLAs, and Product Analysts track business KPIs. Retraining is triggered either bi-weekly or when PSI > 0.05. All issues are logged in Jira with clear escalation paths, and rollbacks are approved by the ML lead.  

This layered monitoring plan provides clear thresholds, ownership, and escalation to ensure that the model remains reliable and valuable after deployment.
