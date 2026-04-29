# Databricks notebook source
# MAGIC %md
# MAGIC # Fleet Repair-or-Replace Decision Demo — Synthetic Data Generator
# MAGIC
# MAGIC Generates a realistic, ML-ready synthetic dataset for a fleet management /
# MAGIC insurance company faced with the question: **when a vehicle is damaged, should
# MAGIC we repair it or replace it?**
# MAGIC
# MAGIC ### About this demo
# MAGIC All data in this notebook is **synthetic**. Vehicle IDs, registration
# MAGIC numbers, driver IDs, prices and labour rates are generated procedurally with
# MAGIC `numpy` + Python's `random` module. No customer or employee data is used.
# MAGIC The notebook is intended for demonstration of Databricks features
# MAGIC (Lakeflow, Unity Catalog, Genie, Mosaic AI) and as a starting point for
# MAGIC building repair-vs-replace classification models.
# MAGIC
# MAGIC ### What gets created
# MAGIC A single Unity Catalog schema with seven Delta tables:
# MAGIC
# MAGIC | # | Table | Rows | Description |
# MAGIC |---|---|---|---|
# MAGIC | 1 | `1_vehicle_master` | ~5,000 | Static vehicle attributes |
# MAGIC | 2 | `2_damage_assessment` | ~12,000 | Damage events and repair estimates |
# MAGIC | 3 | `3_maintenance_history` | ~28,000 | Past services and repairs |
# MAGIC | 4 | `4_incident_details` | ~12,000 | Per-assessment incident narrative + parts breakdown |
# MAGIC | 5 | `5_repair_replace_decisions` | ~12,000 | Historical decision label + downstream outcome |
# MAGIC | 6 | `6_regional_cost_factors` | 9 | Country labour & parts multipliers |
# MAGIC | 7 | `7_parts_catalogue` | ~120 | Parts master with EUR prices and lead times |

# COMMAND ----------

# MAGIC %md ## 1. Configuration
# MAGIC
# MAGIC To run this notebook against a different catalog, edit the `CATALOG`
# MAGIC variable below — everything else is portable.

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "repair_or_replace"

N_VEHICLES = 5000
N_ASSESSMENTS = 12000
SEED = 42

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Writing tables into {CATALOG}.{SCHEMA}")

# COMMAND ----------

import random
import string
import numpy as np
import pandas as pd
from datetime import date, timedelta

random.seed(SEED)
np.random.seed(SEED)

# COMMAND ----------

# MAGIC %md ## 2. Reference data — makes, models, regions, parts
# MAGIC
# MAGIC The fleet is European-skewed (FR/DE/UK/IT/ES/NL/BE/PL/IE) and mixes
# MAGIC passenger vehicles with light commercial vans and a small heavy-truck
# MAGIC tail — typical of a mixed-use fleet operator.

# COMMAND ----------

# (model_name, category, base_engine_cc, primary_fuel, base_purchase_price_eur, depreciation_pct)
MAKES_MODELS = {
    "Toyota":        [("Corolla","Sedan",1600,"Petrol",24000,11.0),
                      ("Camry","Sedan",2000,"Hybrid",32000,10.5),
                      ("RAV4","SUV",2000,"Hybrid",36000,10.0),
                      ("Hilux","Truck",2500,"Diesel",40000,9.0)],
    "Ford":          [("Focus","Sedan",1500,"Petrol",22000,12.5),
                      ("Transit","Van",2200,"Diesel",34000,11.0),
                      ("Fiesta","Sedan",1100,"Petrol",17000,13.0),
                      ("Kuga","SUV",1500,"Diesel",30000,12.0)],
    "Volkswagen":    [("Golf","Sedan",1500,"Petrol",26000,11.0),
                      ("Passat","Sedan",2000,"Diesel",34000,10.5),
                      ("Transporter","Van",2000,"Diesel",38000,10.0),
                      ("Polo","Sedan",1200,"Petrol",18000,12.5)],
    "Mercedes-Benz": [("Sprinter","Van",2200,"Diesel",42000,9.5),
                      ("C-Class","Sedan",1800,"Petrol",45000,12.0),
                      ("Vito","Van",2100,"Diesel",38000,10.0)],
    "BMW":           [("3 Series","Sedan",2000,"Petrol",46000,12.5),
                      ("X3","SUV",2000,"Diesel",52000,12.0),
                      ("5 Series","Sedan",2000,"Diesel",58000,13.0)],
    "Renault":       [("Clio","Sedan",1200,"Petrol",17000,13.5),
                      ("Master","Van",2300,"Diesel",36000,11.0),
                      ("Trafic","Van",2000,"Diesel",30000,11.0),
                      ("Megane","Sedan",1500,"Petrol",22000,13.0)],
    "Peugeot":       [("308","Sedan",1500,"Diesel",24000,12.5),
                      ("Partner","Van",1500,"Diesel",24000,11.5),
                      ("Boxer","Van",2200,"Diesel",34000,10.5)],
    "Fiat":          [("Ducato","Van",2300,"Diesel",36000,10.5),
                      ("Panda","Sedan",1200,"Petrol",14000,14.0),
                      ("500","Sedan",1000,"Petrol",16000,13.0)],
    "Iveco":         [("Daily","Van",2300,"Diesel",40000,10.0)],
    "Volvo":         [("V60","Sedan",2000,"Diesel",40000,11.0),
                      ("XC60","SUV",2000,"Hybrid",48000,11.0),
                      ("FH","Truck",13000,"Diesel",110000,8.5)],
    "Tesla":         [("Model 3","Sedan",0,"Electric",48000,15.0),
                      ("Model Y","SUV",0,"Electric",54000,15.0)],
}

COUNTRIES = ["FR","DE","UK","IT","ES","NL","BE","PL","IE"]
# weights: France/Germany dominant, smaller markets thinner
COUNTRY_WEIGHTS = [0.22,0.20,0.16,0.12,0.10,0.07,0.05,0.05,0.03]

# (country, labour_rate_eur_per_hour, parts_cost_multiplier, avg_lead_time_days)
REGIONAL_COSTS = [
    ("FR", 78,  1.00, 4),
    ("DE", 92,  1.05, 3),
    ("UK", 86,  1.10, 6),
    ("IT", 65,  0.95, 5),
    ("ES", 58,  0.92, 5),
    ("NL", 88,  1.04, 3),
    ("BE", 82,  1.02, 4),
    ("PL", 42,  0.85, 7),
    ("IE", 90,  1.08, 7),
]

# COMMAND ----------

# MAGIC %md ## 3. Table 1 — `1_vehicle_master`
# MAGIC
# MAGIC Static attributes per vehicle. Engine size, depreciation and current market
# MAGIC value are derived from a base price for the model and the vehicle's age
# MAGIC (with noise). EVs get `engine_capacity_cc = 0`.

# COMMAND ----------

today = date(2026, 4, 1)

def vin_like(i):
    # not real VIN format — synthetic identifier
    return f"VH-{2018 + (i % 9)}-{i:06d}"

def gen_plate(country):
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=3))
    tail = "".join(random.choices(string.ascii_uppercase, k=2))
    sep = "-" if country in ("FR","DE","ES") else " "
    return f"{letters}{sep}{digits}{sep}{tail}"

vehicles = []
for i in range(1, N_VEHICLES + 1):
    make = random.choices(list(MAKES_MODELS.keys()),
                          weights=[6,6,6,4,3,5,4,3,2,3,2])[0]
    model, category, base_cc, fuel, base_price, dep_rate = random.choice(MAKES_MODELS[make])
    # engine size jitter — ±200cc within band, 0 stays 0
    engine_cc = 0 if base_cc == 0 else int(np.clip(np.random.normal(base_cc, 100), 800, 14000))
    # vehicle age — fleet right-skewed, mean ~5 years, max 15
    age = float(np.round(np.clip(np.random.gamma(2.2, 2.4), 0.2, 15.0), 1))
    year_of_mfg = today.year - int(age)
    odometer = float(np.round(np.clip(np.random.normal(age * 18000, 12000), 1000, 450000), 0))
    # purchase price — model base ± 10%, EVs and newer get small premium
    purchase_price = float(np.round(base_price * np.random.uniform(0.92, 1.08), 0))
    # current market value — exponential decay using depreciation rate
    market_value = float(np.round(purchase_price * ((1 - dep_rate / 100) ** age) *
                                  np.random.uniform(0.92, 1.05), 0))
    market_value = max(market_value, 500.0)
    transmission = "Automatic" if (fuel in ("Electric","Hybrid") or category in ("Truck","SUV"))\
                  else random.choices(["Manual","Automatic"], weights=[0.55, 0.45])[0]
    insurance_group = int(np.clip(
        np.random.normal({"Sedan":15,"SUV":25,"Van":22,"Truck":35}[category], 6), 1, 50))
    country = random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS)[0]
    vehicles.append({
        "vehicle_id": vin_like(i),
        "registration_plate": gen_plate(country),
        "make": make,
        "model": model,
        "year_of_manufacture": year_of_mfg,
        "vehicle_age_years": age,
        "vehicle_category": category,
        "fuel_type": fuel,
        "engine_capacity_cc": engine_cc,
        "transmission_type": transmission,
        "odometer_km": odometer,
        "purchase_price_eur": purchase_price,
        "current_market_value_eur": market_value,
        "depreciation_rate_pct": dep_rate,
        "insurance_group": insurance_group,
        "country_of_registration": country,
        "in_service_date": today - timedelta(days=int(age * 365)),
        "fleet_segment": random.choice(["Operations","Sales","Logistics","Executive","Pool"]),
    })

vehicles_pdf = pd.DataFrame(vehicles)
print(f"Generated {len(vehicles_pdf):,} vehicles")
vehicles_pdf.head(3)

# COMMAND ----------

(spark.createDataFrame(vehicles_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("1_vehicle_master"))

# COMMAND ----------

# MAGIC %md ## 4. Table 6 — `6_regional_cost_factors` and Table 7 — `7_parts_catalogue`
# MAGIC
# MAGIC Build the small reference tables next so we can use them when costing
# MAGIC damages.

# COMMAND ----------

regional_pdf = pd.DataFrame(REGIONAL_COSTS, columns=[
    "country_code","labour_rate_eur_per_hour","parts_cost_multiplier","avg_lead_time_days"
])
(spark.createDataFrame(regional_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("6_regional_cost_factors"))

# COMMAND ----------

PART_CATEGORIES = [
    # (part_name, category, base_price_eur, base_labour_hours, criticality)
    ("Front bumper",          "Body",        420,  3.5, "Cosmetic"),
    ("Rear bumper",           "Body",        380,  3.0, "Cosmetic"),
    ("Front fender L",        "Body",        220,  2.5, "Cosmetic"),
    ("Front fender R",        "Body",        220,  2.5, "Cosmetic"),
    ("Bonnet / hood",         "Body",        540,  3.0, "Structural"),
    ("Boot lid",              "Body",        460,  2.5, "Structural"),
    ("Roof panel",            "Body",        780,  6.0, "Structural"),
    ("Door front L",          "Body",        520,  3.5, "Structural"),
    ("Door front R",          "Body",        520,  3.5, "Structural"),
    ("Door rear L",           "Body",        480,  3.0, "Structural"),
    ("Door rear R",           "Body",        480,  3.0, "Structural"),
    ("Windscreen",            "Glass",       310,  1.5, "Safety"),
    ("Side window",           "Glass",       180,  1.0, "Safety"),
    ("Rear window",           "Glass",       260,  1.5, "Safety"),
    ("Headlight L",           "Electrical",  280,  1.0, "Safety"),
    ("Headlight R",           "Electrical",  280,  1.0, "Safety"),
    ("Tail light L",          "Electrical",  140,  0.7, "Safety"),
    ("Tail light R",          "Electrical",  140,  0.7, "Safety"),
    ("Side mirror L",         "Electrical",  160,  0.8, "Cosmetic"),
    ("Side mirror R",         "Electrical",  160,  0.8, "Cosmetic"),
    ("Radiator",              "Cooling",     430,  3.0, "Mechanical"),
    ("Intercooler",           "Cooling",     380,  2.5, "Mechanical"),
    ("AC condenser",          "Cooling",     320,  2.0, "Mechanical"),
    ("Engine mount",          "Engine",      210,  3.5, "Structural"),
    ("Cylinder head",         "Engine",     1900, 12.0, "Mechanical"),
    ("Engine block",          "Engine",     6800, 18.0, "Mechanical"),
    ("Turbocharger",          "Engine",     1450,  6.0, "Mechanical"),
    ("Alternator",            "Electrical",  420,  2.0, "Mechanical"),
    ("Starter motor",         "Electrical",  340,  2.0, "Mechanical"),
    ("Battery (12V)",         "Electrical",  180,  0.5, "Routine"),
    ("Battery (HV traction)", "Electrical", 9800, 12.0, "Mechanical"),
    ("Inverter",              "Electrical", 2400,  6.0, "Mechanical"),
    ("Gearbox",               "Drivetrain", 3200, 10.0, "Mechanical"),
    ("Clutch assembly",       "Drivetrain",  680,  6.0, "Mechanical"),
    ("Driveshaft",            "Drivetrain",  520,  4.0, "Mechanical"),
    ("Suspension strut F",    "Suspension",  280,  2.5, "Safety"),
    ("Suspension strut R",    "Suspension",  260,  2.5, "Safety"),
    ("Wheel hub",             "Suspension",  220,  2.0, "Safety"),
    ("Brake disc",            "Brakes",      120,  1.0, "Safety"),
    ("Brake caliper",         "Brakes",      210,  1.5, "Safety"),
    ("Brake pads (set)",      "Brakes",       80,  0.8, "Routine"),
    ("ABS module",            "Brakes",      640,  2.5, "Safety"),
    ("Airbag (driver)",       "Safety",      720,  2.0, "Safety"),
    ("Airbag (passenger)",    "Safety",      680,  2.0, "Safety"),
    ("Seat belt assembly",    "Safety",      180,  1.0, "Safety"),
    ("Wheel + tyre",          "Wheels",      280,  0.5, "Routine"),
    ("Tyre only",             "Wheels",      140,  0.4, "Routine"),
    ("Exhaust system",        "Exhaust",     520,  2.5, "Mechanical"),
    ("Catalytic converter",   "Exhaust",     820,  2.0, "Mechanical"),
    ("DPF (diesel)",          "Exhaust",    1200,  3.0, "Mechanical"),
]

parts_rows = []
pid = 1
for name, cat, price, hours, crit in PART_CATEGORIES:
    # variability: factor of 0.85–1.2 across "OEM equivalents"
    for variant in range(random.choice([2, 3])):
        parts_rows.append({
            "part_id": f"P-{pid:05d}",
            "part_name": name,
            "part_category": cat,
            "base_price_eur": float(round(price * np.random.uniform(0.85, 1.2), 2)),
            "labour_hours_estimate": float(round(hours * np.random.uniform(0.9, 1.1), 1)),
            "criticality": crit,
            "lead_time_days_eu_avg": int(np.clip(np.random.normal(5, 3), 1, 30)),
            "oem_supplier": random.choice(["Bosch","Continental","Valeo","Magna","ZF","Denso","Mahle","Brembo"]),
        })
        pid += 1

parts_pdf = pd.DataFrame(parts_rows)
print(f"Parts catalogue rows: {len(parts_pdf)}")
(spark.createDataFrame(parts_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("7_parts_catalogue"))

# COMMAND ----------

# MAGIC %md ## 5. Table 2 — `2_damage_assessment`
# MAGIC
# MAGIC Each row is a single damage event. We pick a random vehicle, generate an
# MAGIC incident type, and synthesise a damage severity distribution. Repair cost
# MAGIC is a function of severity, vehicle category and country labour rate. We
# MAGIC also store a `repair_cost_ratio` — the canonical feature used downstream
# MAGIC by repair-vs-replace models.

# COMMAND ----------

# severity → (repair_cost_floor, repair_cost_ceiling_as_pct_of_market_value, prob_total_loss)
SEVERITY_PROFILE = {
    "Minor":    (200,  0.15, 0.00),
    "Moderate": (800,  0.45, 0.03),
    "Severe":   (2500, 0.90, 0.18),
    "Critical": (6000, 1.80, 0.55),
}
INCIDENT_TYPES = [
    ("Collision",       0.42),
    ("Mechanical",      0.22),
    ("Theft/Vandalism", 0.10),
    ("Weather",         0.08),
    ("Flood",           0.04),
    ("Fire",            0.03),
    ("Wear & tear",     0.11),
]
INCIDENT_NAMES = [t for t,_ in INCIDENT_TYPES]
INCIDENT_PROBS = [p for _,p in INCIDENT_TYPES]

DAMAGE_AREAS = ["Front-end","Rear-end","Side L","Side R","Roof","Underbody",
                "Engine bay","Cabin","Multi-area"]

# vehicles indexed for fast picking
veh_array = vehicles_pdf.to_dict("records")

assessments = []
# some vehicles are assessed multiple times (5–10% are repeats)
vehicle_pool = [v["vehicle_id"] for v in veh_array]
vehicle_pool += random.sample(vehicle_pool, k=int(0.20 * N_VEHICLES))
random.shuffle(vehicle_pool)

for i in range(1, N_ASSESSMENTS + 1):
    v = veh_array[random.randrange(len(veh_array))] if i > len(vehicle_pool) \
        else next(x for x in veh_array if x["vehicle_id"] == vehicle_pool[i-1])

    # severity weighting depends loosely on age — older vehicles see more severe events
    age = v["vehicle_age_years"]
    sev_weights = np.array([0.55, 0.30, 0.12, 0.03])
    if age > 8:
        sev_weights = sev_weights + np.array([-0.15, 0.00, 0.10, 0.05])
    if age > 12:
        sev_weights = sev_weights + np.array([-0.15, -0.05, 0.10, 0.10])
    sev_weights = np.clip(sev_weights, 0.01, 1.0)
    sev_weights = sev_weights / sev_weights.sum()
    severity = np.random.choice(["Minor","Moderate","Severe","Critical"], p=sev_weights)

    incident = np.random.choice(INCIDENT_NAMES, p=INCIDENT_PROBS)
    floor, ceil_ratio, _ = SEVERITY_PROFILE[severity]

    # repair cost — uniform between floor and ceiling*market_value, with category multiplier
    cat_mult = {"Sedan":1.0,"SUV":1.15,"Van":1.10,"Truck":1.30}[v["vehicle_category"]]
    base_cost = np.random.uniform(floor, max(floor + 300, ceil_ratio * v["current_market_value_eur"]))
    base_cost *= cat_mult

    # apply country labour multiplier
    rcost_row = next(r for r in REGIONAL_COSTS if r[0] == v["country_of_registration"])
    labour_rate = rcost_row[1]
    parts_mult = rcost_row[2]
    # split rough: 60% parts, 40% labour
    parts_cost = float(round(base_cost * 0.60 * parts_mult, 2))
    labour_hours = float(round(base_cost * 0.40 / 80.0, 1))  # normalised hours
    labour_cost = float(round(labour_hours * labour_rate, 2))
    estimated_repair_cost = float(round(parts_cost + labour_cost, 2))

    # repair cost as ratio of market value — the headline feature
    repair_ratio = round(estimated_repair_cost / max(v["current_market_value_eur"], 1.0), 3)

    # downtime in days — bigger on severe / critical / parts heavy
    downtime = int(np.clip(np.random.normal(
        {"Minor":3,"Moderate":8,"Severe":18,"Critical":35}[severity], 4), 1, 90))

    # date — within last 24 months
    delta_days = random.randint(0, 730)
    a_date = today - timedelta(days=delta_days)

    assessments.append({
        "assessment_id": f"ASS-{a_date.year}-{i:06d}",
        "vehicle_id": v["vehicle_id"],
        "assessment_date": a_date,
        "incident_type": incident,
        "damage_severity": severity,
        "primary_damage_area": random.choice(DAMAGE_AREAS),
        "estimated_repair_cost_eur": estimated_repair_cost,
        "estimated_parts_cost_eur": parts_cost,
        "estimated_labour_cost_eur": labour_cost,
        "estimated_labour_hours": labour_hours,
        "estimated_downtime_days": downtime,
        "current_market_value_eur": v["current_market_value_eur"],
        "repair_cost_ratio": float(repair_ratio),
        "vehicle_age_at_incident_years": v["vehicle_age_years"],
        "odometer_at_incident_km": v["odometer_km"],
        "country_of_registration": v["country_of_registration"],
        "assessor_id": f"ASR-{random.randint(1, 120):04d}",
        "assessor_region": random.choice(["North","South","East","West","Central"]),
    })

assessments_pdf = pd.DataFrame(assessments)
print(f"Assessments: {len(assessments_pdf):,}, "
      f"mean repair ratio: {assessments_pdf['repair_cost_ratio'].mean():.2f}")
(spark.createDataFrame(assessments_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("2_damage_assessment"))

# COMMAND ----------

# MAGIC %md ## 6. Table 3 — `3_maintenance_history`
# MAGIC
# MAGIC Routine + unscheduled maintenance prior to / between damage assessments.
# MAGIC Older / higher-mileage vehicles get more service entries — the
# MAGIC `prior_maintenance_count` derived from this table is a strong predictor
# MAGIC of replace decisions.

# COMMAND ----------

SERVICE_TYPES = [
    ("Oil change",             "Routine",   80,   1.0),
    ("Brake pads",             "Routine",   180,  1.5),
    ("Tyre replacement",       "Routine",   260,  0.8),
    ("Battery replacement",    "Routine",   200,  0.7),
    ("Annual inspection",      "Routine",   120,  2.0),
    ("Timing belt",            "Major",     520,  6.0),
    ("Clutch overhaul",        "Major",     820,  6.5),
    ("Transmission service",   "Major",     680,  4.5),
    ("Engine diagnostic",      "Diagnostic",110,  1.5),
    ("DPF regeneration",       "Diagnostic",240,  2.0),
    ("Suspension overhaul",    "Major",     940,  7.0),
    ("Air conditioning",       "Routine",   180,  1.5),
    ("Software update",        "Diagnostic",90,   1.0),
]

maintenance_rows = []
maint_id = 1
for v in veh_array:
    age = v["vehicle_age_years"]
    # baseline ~2 routine services / year of age, plus extra for older vehicles
    expected = max(1, int(age * 2 + np.random.poisson(1.5)))
    if v["odometer_km"] > 200000:
        expected += 4
    for _ in range(expected):
        s = random.choices(SERVICE_TYPES,
                           weights=[6,4,4,3,5,1,1,2,3,1,1,2,1])[0]
        name, kind, base_cost, hours = s
        rcost_row = next(r for r in REGIONAL_COSTS if r[0] == v["country_of_registration"])
        labour_rate, parts_mult, _ = rcost_row[1], rcost_row[2], rcost_row[3]
        parts = float(round(base_cost * 0.55 * parts_mult * np.random.uniform(0.85, 1.15), 2))
        labour = float(round(hours * labour_rate * np.random.uniform(0.9, 1.1), 2))
        cost = round(parts + labour, 2)
        days_back = random.randint(30, max(60, int(age * 365)))
        sdate = today - timedelta(days=days_back)
        maintenance_rows.append({
            "maintenance_id": f"MAINT-{maint_id:07d}",
            "vehicle_id": v["vehicle_id"],
            "service_date": sdate,
            "service_type": name,
            "service_category": kind,
            "parts_cost_eur": parts,
            "labour_cost_eur": labour,
            "total_cost_eur": cost,
            "odometer_at_service_km": float(np.clip(
                v["odometer_km"] - np.random.uniform(0, v["odometer_km"]*0.6), 100, v["odometer_km"])),
            "garage_location": random.choice(["Dealer","Independent","Fleet workshop","Mobile"]),
            "country": v["country_of_registration"],
        })
        maint_id += 1

maintenance_pdf = pd.DataFrame(maintenance_rows)
print(f"Maintenance records: {len(maintenance_pdf):,}")
(spark.createDataFrame(maintenance_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("3_maintenance_history"))

# COMMAND ----------

# MAGIC %md ## 7. Table 4 — `4_incident_details`
# MAGIC
# MAGIC One row per assessment. Captures incident narrative, third-party / driver
# MAGIC info, and a stringified JSON-ish list of impacted parts (good fodder for
# MAGIC LLM-based extraction demos with Mosaic AI / `ai_query`).

# COMMAND ----------

NARRATIVE_TEMPLATES = {
    "Collision":       ["Rear-ended at traffic light by {3p}",
                        "Side-swipe collision in roundabout near {city}",
                        "Front impact with stationary object during {weather} conditions",
                        "Multi-vehicle pile-up on {road} motorway",
                        "Low-speed parking lot collision with {3p}"],
    "Mechanical":      ["Engine warning light followed by power loss on {road}",
                        "Transmission failure after odometer reached high mileage",
                        "Cooling system failure leading to overheating",
                        "Drivetrain noise diagnosed as failing differential",
                        "Loss of braking pressure noticed by driver"],
    "Theft/Vandalism": ["Forced entry through driver door, console damaged",
                        "Vehicle vandalised overnight in unsecured lot",
                        "Catalytic converter cut from underside",
                        "Side mirrors and badges stolen",
                        "Recovered after theft with significant damage"],
    "Weather":         ["Hailstorm damage across roof and bonnet",
                        "Wind-blown debris struck windscreen",
                        "Heavy rain led to aquaplane and barrier impact",
                        "Sub-zero conditions caused brake line damage"],
    "Flood":           ["Submerged in flash flood up to door line",
                        "Engine ingested water during river crossing",
                        "Standing water inside cabin after roof leak"],
    "Fire":            ["Engine bay fire — total loss inspection requested",
                        "Electrical fire originating from battery compartment",
                        "External fire spread from adjacent vehicle"],
    "Wear & tear":     ["Suspension components worn beyond service limit",
                        "Multiple advisories at MOT — accumulated repairs needed",
                        "Heavy corrosion under chassis identified at inspection"],
}
CITIES = ["Lyon","Hamburg","Birmingham","Milan","Madrid","Rotterdam","Brussels","Wroclaw","Cork"]
ROADS  = ["A1","A6","M1","E40","A4","E15","M50"]
WEATHER = ["clear","wet","icy","foggy","heavy rain"]
THIRD_PARTIES = ["a passenger car","a delivery van","a HGV","a motorcycle","a stationary vehicle"]

def render_narrative(incident):
    tpl = random.choice(NARRATIVE_TEMPLATES[incident])
    return tpl.format(
        city=random.choice(CITIES),
        road=random.choice(ROADS),
        weather=random.choice(WEATHER),
        **{"3p": random.choice(THIRD_PARTIES)},
    )

# parts list per assessment — pulled from parts catalogue and scaled by severity
parts_by_cat = parts_pdf.groupby("part_category")["part_id"].apply(list).to_dict()
SEV_PARTS = {"Minor":(1,3),"Moderate":(3,6),"Severe":(5,10),"Critical":(8,16)}

incident_rows = []
for a in assessments:
    n_parts_lo, n_parts_hi = SEV_PARTS[a["damage_severity"]]
    n_parts = random.randint(n_parts_lo, n_parts_hi)
    chosen_parts = random.sample(parts_pdf["part_id"].tolist(), k=min(n_parts, len(parts_pdf)))
    incident_rows.append({
        "assessment_id": a["assessment_id"],
        "vehicle_id": a["vehicle_id"],
        "incident_date": a["assessment_date"] - timedelta(days=random.randint(0, 6)),
        "incident_type": a["incident_type"],
        "narrative": render_narrative(a["incident_type"]),
        "third_party_involved": a["incident_type"] == "Collision" and random.random() < 0.6,
        "police_report_filed": a["damage_severity"] in ("Severe","Critical") or random.random() < 0.2,
        "weather_condition": random.choice(WEATHER),
        "location_country": a["country_of_registration"],
        "location_road": random.choice(ROADS),
        "driver_id": f"DRV-{random.randint(1, 1500):05d}",
        "impacted_part_ids": ",".join(chosen_parts),
        "n_impacted_parts": len(chosen_parts),
        "airbag_deployed": a["damage_severity"] in ("Severe","Critical") and random.random() < 0.4,
    })

incident_pdf = pd.DataFrame(incident_rows)
(spark.createDataFrame(incident_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("4_incident_details"))

# COMMAND ----------

# MAGIC %md ## 8. Table 5 — `5_repair_replace_decisions`
# MAGIC
# MAGIC The labelled outcome table — what the assessor actually decided, plus the
# MAGIC realised downstream outcome 6–12 months later. The label `decision`
# MAGIC is the supervised target for ML, while `decision_was_optimal` is a
# MAGIC retrospective quality flag.
# MAGIC
# MAGIC ### Decision logic (with realistic noise)
# MAGIC
# MAGIC * `repair_cost_ratio > 0.75` → tend to **Replace**
# MAGIC * `repair_cost_ratio > 0.55` AND `vehicle_age > 10` → tend to **Replace**
# MAGIC * `damage_severity == "Critical"` → almost always **Replace**
# MAGIC * `vehicle_age > 13` → tilt towards **Replace**
# MAGIC * Otherwise → **Repair**
# MAGIC
# MAGIC On top of this baseline, ~12% of decisions are flipped to introduce
# MAGIC realistic assessor inconsistency — which is the *whole point* of the
# MAGIC use case (the dataset has to reflect inconsistent human decisions for ML
# MAGIC to add value).

# COMMAND ----------

# join assessment with vehicle for decision logic
assess_with_veh = assessments_pdf.merge(
    vehicles_pdf[["vehicle_id","vehicle_age_years","current_market_value_eur"]],
    on="vehicle_id", suffixes=("","_v"))

decision_rows = []
optimal_match = 0
for _, a in assess_with_veh.iterrows():
    age = a["vehicle_age_years"]
    sev = a["damage_severity"]
    ratio = a["repair_cost_ratio"]

    # baseline economic decision
    if sev == "Critical":
        baseline = "Replace"
    elif ratio > 0.75:
        baseline = "Replace"
    elif ratio > 0.55 and age > 10:
        baseline = "Replace"
    elif age > 13:
        baseline = "Replace" if random.random() < 0.6 else "Repair"
    else:
        baseline = "Repair"

    # ~12% of assessors flip the decision (regional bias / inconsistency)
    flipped = random.random() < 0.12
    actual = ("Repair" if baseline == "Replace" else "Replace") if flipped else baseline

    # downstream outcome
    repair_completed_in_days = (
        int(np.clip(np.random.normal(a["estimated_downtime_days"]*1.2, 5), 1, 120))
        if actual == "Repair" else None
    )
    repair_actual_cost = (
        float(round(a["estimated_repair_cost_eur"] * np.random.uniform(0.92, 1.18), 2))
        if actual == "Repair" else None
    )
    # post-repair recurrence (came back for related issue within 6 months)
    post_repair_recurrence = (
        actual == "Repair" and (sev in ("Severe","Critical") or age > 9) and random.random() < 0.18
    )
    salvage_value = (
        float(round(a["current_market_value_eur"] * np.random.uniform(0.05, 0.20), 2))
        if actual == "Replace" else None
    )
    replacement_cost = (
        float(round(a["current_market_value_eur"] * np.random.uniform(0.95, 1.15) +
                    np.random.uniform(800, 2500), 2))
        if actual == "Replace" else None
    )

    optimal = (actual == baseline)
    optimal_match += int(optimal)

    decision_rows.append({
        "decision_id": f"DEC-{a['assessment_id'].replace('ASS-','')}",
        "assessment_id": a["assessment_id"],
        "vehicle_id": a["vehicle_id"],
        "decision_date": a["assessment_date"] + timedelta(days=random.randint(0, 5)),
        "decision": actual,
        "decision_method": random.choice(["Manual","Manual","Manual","Rule-based"]),
        "baseline_recommendation": baseline,
        "decision_was_optimal": optimal,
        "approver_id": f"APR-{random.randint(1, 80):04d}",
        "approval_region": random.choice(["North","South","East","West","Central"]),
        "repair_actual_cost_eur": repair_actual_cost,
        "repair_completed_in_days": repair_completed_in_days,
        "post_repair_recurrence_within_6m": post_repair_recurrence,
        "salvage_value_eur": salvage_value,
        "replacement_cost_eur": replacement_cost,
    })

decisions_pdf = pd.DataFrame(decision_rows)
print(f"Decisions: {len(decisions_pdf):,}, "
      f"replace rate: {(decisions_pdf['decision']=='Replace').mean():.1%}, "
      f"baseline-actual agreement: {decisions_pdf['decision_was_optimal'].mean():.1%}")
(spark.createDataFrame(decisions_pdf)
      .write.mode("overwrite").option("overwriteSchema","true")
      .saveAsTable("5_repair_replace_decisions"))

# COMMAND ----------

# MAGIC %md ## 9. Validation
# MAGIC
# MAGIC Quick sanity checks: row counts, FK integrity, label balance and
# MAGIC distribution of repair ratios by decision.

# COMMAND ----------

display(spark.sql("""
SELECT
  (SELECT COUNT(*) FROM 1_vehicle_master)             AS vehicles,
  (SELECT COUNT(*) FROM 2_damage_assessment)          AS assessments,
  (SELECT COUNT(*) FROM 3_maintenance_history)        AS maintenance,
  (SELECT COUNT(*) FROM 4_incident_details)           AS incidents,
  (SELECT COUNT(*) FROM 5_repair_replace_decisions)   AS decisions,
  (SELECT COUNT(*) FROM 6_regional_cost_factors)      AS regions,
  (SELECT COUNT(*) FROM 7_parts_catalogue)            AS parts
"""))

# COMMAND ----------

display(spark.sql("""
SELECT
  decision,
  COUNT(*) AS n,
  ROUND(AVG(d.repair_actual_cost_eur), 0) AS avg_actual_repair_cost,
  ROUND(AVG(a.repair_cost_ratio), 2)      AS avg_repair_cost_ratio,
  ROUND(AVG(a.vehicle_age_at_incident_years), 1) AS avg_age,
  ROUND(AVG(CASE WHEN d.post_repair_recurrence_within_6m THEN 1 ELSE 0 END), 3) AS recurrence_rate
FROM 5_repair_replace_decisions d
JOIN 2_damage_assessment a USING (assessment_id)
GROUP BY decision
"""))

# COMMAND ----------

display(spark.sql("""
SELECT damage_severity, decision, COUNT(*) AS n
FROM 2_damage_assessment a
JOIN 5_repair_replace_decisions d USING (assessment_id)
GROUP BY damage_severity, decision
ORDER BY damage_severity, decision
"""))

# COMMAND ----------

# MAGIC %md ## 10. Done
# MAGIC
# MAGIC Tables are written to `{CATALOG}.{SCHEMA}` and ready for:
# MAGIC * **Genie** — connect a Genie room to all 7 tables for natural-language Q&A
# MAGIC * **AutoML / Mosaic AI** — train a binary classifier on
# MAGIC   `5_repair_replace_decisions.decision` with features from the joined
# MAGIC   vehicle/assessment/maintenance tables
# MAGIC * **AI Functions / `ai_query`** — extract structured data from
# MAGIC   `4_incident_details.narrative` using Claude on the Foundation Model API
# MAGIC * **Lakeview dashboards** — build cost dashboards by region/make/severity
