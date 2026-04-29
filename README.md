# Fleet Repair-or-Replace Decision Demo

Synthetic dataset for a fleet management / insurance company facing the
question: **when a vehicle is damaged, should we repair it or replace it?**

Today the decision is made manually by assessors — leading to inconsistent
outcomes across regions, suboptimal cost choices, fleet-availability delays,
and bias. This dataset is built to demonstrate how Databricks (Unity Catalog,
Genie, Mosaic AI / AutoML, AI Functions) can drive that decision consistently
and explain it.

> **About this demo** — every record is synthetically generated. Vehicle IDs,
> registration plates, driver IDs, prices and labour rates are all produced
> procedurally with `numpy` and Python's `random` module. There is no
> customer, employee or third-party data of any kind.

---

## What gets built

A single Unity Catalog schema with seven Delta tables.

| # | Table | Rows | Purpose |
|---|---|---|---|
| 1 | `1_vehicle_master` | ~5,000 | Vehicle static attributes — make, model, age, fuel, market value |
| 2 | `2_damage_assessment` | ~12,000 | Damage events with severity, repair-cost estimate and the headline `repair_cost_ratio` feature |
| 3 | `3_maintenance_history` | ~28,000 | Routine + unscheduled service history per vehicle |
| 4 | `4_incident_details` | ~12,000 | Incident narratives + impacted-parts list (good fodder for `ai_query` extraction demos) |
| 5 | `5_repair_replace_decisions` | ~12,000 | **Labelled decisions** with downstream outcomes (recurrence, salvage, replacement cost) |
| 6 | `6_regional_cost_factors` | 9 | Country labour rate, parts cost multiplier, lead time |
| 7 | `7_parts_catalogue` | ~120 | Parts master with EUR prices, OEM supplier and lead times |

### Headline numbers

* **Fleet:** 5,000 vehicles, 11 makes, 30+ models, 9 European countries
* **Mix:** Sedans, SUVs, Vans, Trucks; Petrol / Diesel / Hybrid / Electric
* **Damage events:** 12,000 covering 24 months
* **Incident types:** Collision, Mechanical, Theft/Vandalism, Weather, Flood, Fire, Wear & tear
* **Severity bands:** Minor / Moderate / Severe / Critical
* **Decision label balance:** ~60% Repair / ~40% Replace (by design — varies with age and severity)
* **Assessor inconsistency baked in:** ~12% of decisions diverge from the rule-based baseline — exactly the noise an ML model should learn to clean up

---

## Decision logic (with deliberate noise)

Each historical decision in `5_repair_replace_decisions` is generated from a
realistic baseline rule, then ~12% are flipped to mimic real-world assessor
inconsistency:

* `repair_cost_ratio > 0.75` → **Replace**
* `repair_cost_ratio > 0.55` AND `vehicle_age > 10` → **Replace**
* `damage_severity = "Critical"` → almost always **Replace**
* `vehicle_age > 13` → tilt towards **Replace**
* Otherwise → **Repair**

The flipped ~12% are flagged via `decision_was_optimal = false` — useful for
demonstrating "where the human got it wrong" in dashboards.

---

## How to run

### Quick path — Databricks workspace

1. Edit `notebooks/00_generate_repair_or_replace_data.py` and set `CATALOG`
   to your Unity Catalog (default: `lr_serverless_aws_us_catalog`).
2. Import the notebook and run all cells on **serverless compute**. No
   cluster libraries needed — the notebook uses only `numpy` and `pandas`,
   which are pre-installed.
3. Tables land in `<catalog>.repair_or_replace.*`.

### Portability

To target a different catalog from the command line:

```bash
sed -i '' 's/lr_serverless_aws_us_catalog/your_catalog/g' \
  notebooks/00_generate_repair_or_replace_data.py
```

---

## What you can do next

* **Genie** — Point a Genie room at all 7 tables for natural-language
  questions like "which makes have the worst recurrence rate after repair?"
* **AutoML / Mosaic AI** — Train a binary classifier on
  `5_repair_replace_decisions.decision`, joining features from
  `1_vehicle_master`, `2_damage_assessment` and `3_maintenance_history`.
* **AI Functions** — Use `ai_query` with Claude (Foundation Model API) to
  extract structured fields from `4_incident_details.narrative`.
* **Lakeview** — Build a cost-by-region / cost-by-severity dashboard.
* **Vector Search** — Index incident narratives to find similar past cases
  for any new claim.

---

## Repo layout

```
.
├── README.md
└── notebooks/
    └── 00_generate_repair_or_replace_data.py   # single-notebook generator
```

---

## License

MIT — synthetic data, do whatever you want with it.
