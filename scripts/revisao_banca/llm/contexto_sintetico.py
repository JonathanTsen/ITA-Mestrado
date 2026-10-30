"""Experimento exploratório pedido pelo autor em 30/10/2026: o contexto do domínio ajuda o LLM a classificar o mecanismo?

Oito bases sintéticas inventadas agora (o LLM não pode tê-las visto), cada uma com um contexto detalhado: para que
serve, quem coletou, como, o que é cada linha, o significado, a unidade e a faixa típica de cada coluna (as faixas saem
dos próprios dados completos) e as relações conhecidas entre as variáveis. O contexto nunca fala de dados faltantes e é
o mesmo para os três mecanismos de um trio. O apagamento é o do benchmark: a etapa `inject` com o gerador simétrico
(trios com a mesma contagem de ausências), sobre estas tabelas.

Braços, todos com as mesmas linhas e a mesma ordem de colunas da tabela cega do A1:
- S0, cego: exatamente o A1 da produção (colunas c0…ck, inteiros de 0 a 999), sem contexto;
- S1, só os nomes reais das colunas, valores em 0–999;
- S2, nomes + contexto detalhado, valores em 0–999;
- S3, nomes + contexto detalhado + valores originais (3 algarismos significativos).

Uso (no ITA-Mestrado; o Flash está autorizado, o Argon não roda aqui):
  uv run --frozen python scripts/revisao_banca/llm/contexto_sintetico.py gerar data_rb/contexto_sintetico
  uv run --frozen python -m missdetect.rb.cli inject --config configs/rb/contexto-sintetico.yaml
  uv run --frozen python scripts/revisao_banca/llm/contexto_sintetico.py perguntar <run> [--budget-only]
  uv run --frozen python scripts/revisao_banca/llm/contexto_sintetico.py analisar <run> <diretório de saída>
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from missdetect.rb.config import load_config
from missdetect.rb.evaluate import metrics, summarize_by_family
from missdetect.rb.ids import child_seed
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.llm.cache import CLASSES, ResponseCache
from missdetect.rb.llm.client import GeminiBackend, Runner, build_request, load_api_keys
from missdetect.rb.llm.render import (
    A1_SCHEMA,
    A1_SYSTEM,
    A3_SYSTEM,
    MAX_ROWS,
    NA_TOKEN,
    blind_table,
    render_a1,
)
from missdetect.rb.llm.run import budget_gate, build_panel, cache_root, predictions_from_cache
from missdetect.rb.runs import read_observed, repo_root
from missdetect.rb.stats import bootstrap_hl_ci, hodges_lehmann

MODEL = "gemini-3.8-flash"
GENERATOR_SEED = 20261030
N_ROWS = 1000
RENDERER = "contexto-sintetico-1"
ARMS = ("S0", "S1", "S2", "S3")

_SCALED = "Each column was rescaled by an increasing linear transformation and rounded to integers from 0 to 999."
_RAW = "The values are the original ones, rounded to 3 significant digits."
S2_SYSTEM = A3_SYSTEM.replace("a short description", "a detailed description")
S1_SYSTEM = S2_SYSTEM.replace(
    "You will receive a detailed description of a dataset and a table from it in CSV format.",
    "You will receive a table from a dataset in CSV format.",
)
S3_SYSTEM = S2_SYSTEM.replace(_SCALED, _RAW)
assert len({A3_SYSTEM, S1_SYSTEM, S2_SYSTEM, S3_SYSTEM}) == 4, "as instruções dos braços precisam diferir"


@dataclass(frozen=True)
class Domain:
    source_id: int
    name: str
    story: str
    columns: dict[str, str]  # nome → significado e unidade
    relationships: str
    generate: Callable[[np.random.Generator, int], pd.DataFrame]


def _clinic(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = rng.uniform(20, 85, n)
    bmi = np.clip(rng.normal(27 + 0.04 * (age - 50), 4.5), 16, 50)
    activity = np.exp(rng.normal(5.0 - 0.03 * (bmi - 27), 0.5))
    sleep = np.clip(rng.normal(7.2 - 0.01 * (age - 50) - 0.03 * (bmi - 27), 0.9), 3.5, 11)
    sbp = 95 + 0.5 * age + 0.9 * (bmi - 25) + rng.normal(0, 11, n)
    glucose = 78 + 1.9 * (bmi - 25) + 0.35 * (age - 40) - 0.02 * activity + np.exp(rng.normal(2.3, 0.6, n))
    hba1c = 3.2 + 0.022 * glucose + rng.normal(0, 0.35, n)
    ldl = 95 + 0.45 * age + 1.2 * (bmi - 25) + rng.normal(0, 25, n)
    return pd.DataFrame({
        "age_years": age.round(0), "bmi_kg_m2": bmi.round(1), "activity_min_week": activity.round(0),
        "sleep_hours": sleep.round(1), "systolic_bp_mmhg": sbp.round(0), "fasting_glucose_mg_dl": glucose.round(0),
        "hba1c_pct": hba1c.round(1), "ldl_cholesterol_mg_dl": ldl.round(0),
    })  # fmt: skip


def _energy(rng: np.random.Generator, n: int) -> pd.DataFrame:
    income = np.exp(rng.normal(4.0, 0.45, n))
    area = np.clip(35 + 0.7 * income + rng.normal(0, 18, n), 25, 400)
    temp = rng.normal(26, 3, n)
    ac_kw = np.clip(0.6 + 0.025 * area + 0.15 * (temp - 24) + rng.normal(0, 0.6, n), 0.2, None)
    appliance_age = rng.gamma(2.0, 3.0, n)
    solar = np.exp(rng.normal(3.3, 0.9, n))
    kwh = np.clip(
        70 + 2.0 * area + 5.5 * ac_kw * np.clip(temp - 20, 0, None) + 4 * appliance_age - 0.6 * solar
        + rng.normal(0, 45, n), 30, None,
    )  # fmt: skip
    bill = 12 + kwh * np.clip(rng.normal(0.78, 0.1, n), 0.4, None) + rng.normal(0, 10, n)
    return pd.DataFrame({
        "household_income_k": income.round(1), "floor_area_m2": area.round(0), "outdoor_temp_c": temp.round(1),
        "ac_capacity_kw": ac_kw.round(2), "appliance_age_years": appliance_age.round(1),
        "solar_kwh_month": solar.round(0), "monthly_kwh": kwh.round(0), "monthly_bill": bill.round(1),
    })  # fmt: skip


def _farm(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rainfall = np.clip(rng.normal(650, 160, n), 200, None)
    irrigation = rng.gamma(3.0, 9.0, n)
    fertilizer = np.clip(rng.normal(120, 40, n), 10, None)
    ph = rng.normal(6.3, 0.55, n)
    temp = rng.normal(24, 2.5, n)
    pest = np.clip(rng.normal(30 + 1.5 * (temp - 24), 9, n), 0, None)
    area = np.exp(rng.normal(0.3, 0.6, n))
    crop = (0.9 + 0.0035 * rainfall + 0.012 * irrigation + 0.009 * fertilizer - 0.9 * (ph - 6.5) ** 2
            - 0.06 * (temp - 24) ** 2 - 0.025 * pest + rng.normal(0, 0.45, n))  # fmt: skip
    return pd.DataFrame({
        "season_rainfall_mm": rainfall.round(0), "irrigation_hours": irrigation.round(1),
        "fertilizer_kg_ha": fertilizer.round(0), "soil_ph": ph.round(2), "mean_temp_c": temp.round(1),
        "pest_index": pest.round(1), "plot_area_ha": area.round(2), "yield_t_ha": crop.round(2),
    })  # fmt: skip


def _grocery(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = rng.uniform(18, 80, n)
    distance = np.exp(rng.normal(1.6, 0.6, n))
    sessions = np.clip(rng.normal(9 - 0.05 * (age - 40), 3.5), 0.5, None)
    minutes = np.clip(rng.normal(14 + 0.08 * (age - 40), 4), 1, None)
    discount = np.clip(rng.normal(8, 4, n), 0, 40)
    delivery = np.clip(rng.normal(3 + 0.9 * distance, 1.5), 0.2, None)
    spend = np.clip(40 + 9 * sessions + 2.5 * minutes + 1.5 * discount + rng.normal(0, 30, n), 5, None)
    satisfaction = np.clip(8.6 - 0.35 * delivery + 0.05 * discount + rng.normal(0, 0.8, n), 0, 10)
    return pd.DataFrame({
        "customer_age": age.round(0), "distance_km": distance.round(1), "sessions_per_month": sessions.round(1),
        "avg_session_minutes": minutes.round(1), "discount_pct": discount.round(1), "delivery_hours": delivery.round(1),
        "monthly_spend": spend.round(1), "satisfaction_score": satisfaction.round(1),
    })  # fmt: skip


def _extrusion(rng: np.random.Generator, n: int) -> pd.DataFrame:
    hours = rng.uniform(0, 720, n)
    speed = rng.normal(85, 12, n)
    humidity = rng.normal(55, 12, n)
    temp = 205 + 0.04 * speed + 0.01 * hours + rng.normal(0, 4, n)
    vibration = np.clip(1.2 + 0.004 * hours + 0.01 * (speed - 85) + rng.normal(0, 0.45, n), 0.1, None)
    pressure = 140 + 0.6 * speed - 0.5 * (temp - 210) + rng.normal(0, 8, n)
    power = 18 + 0.25 * speed + 0.04 * pressure + rng.normal(0, 2.5, n)
    defects = np.exp(
        1.0 + 0.45 * vibration + 0.06 * np.abs(temp - 212) + 0.01 * (humidity - 55) + rng.normal(0, 0.35, n)
    )
    return pd.DataFrame({
        "hours_since_maintenance": hours.round(0), "screw_speed_rpm": speed.round(1),
        "ambient_humidity_pct": humidity.round(1), "barrel_temp_c": temp.round(1), "vibration_mm_s": vibration.round(2),
        "melt_pressure_bar": pressure.round(1), "motor_power_kw": power.round(1), "defects_per_1000": defects.round(1),
    })  # fmt: skip


def _school(rng: np.random.Generator, n: int) -> pd.DataFrame:
    income = np.exp(rng.normal(3.9, 0.5, n))
    prior = np.clip(rng.normal(62 + 4 * np.log(income / 50), 12), 10, 100)
    study = np.clip(rng.normal(8 + 0.06 * (prior - 60), 3.5), 0, None)
    attendance = np.clip(rng.normal(90 + 0.08 * (prior - 60), 6), 40, 100)
    commute = np.exp(rng.normal(3.0, 0.5, n))
    sleep = np.clip(rng.normal(7.8 - 0.012 * (commute - 20), 0.9), 4, 11)
    extra = rng.gamma(2.0, 2.0, n)
    final = np.clip(15 + 0.55 * prior + 1.1 * study + 0.25 * (attendance - 90) - 0.08 * commute + 1.2 * (sleep - 7.5)
                    + rng.normal(0, 7, n), 0, 100)  # fmt: skip
    return pd.DataFrame({
        "household_income_k": income.round(1), "prior_year_score": prior.round(1), "study_hours_week": study.round(1),
        "attendance_pct": attendance.round(1), "commute_minutes": commute.round(0), "sleep_hours": sleep.round(1),
        "extracurricular_hours": extra.round(1), "final_exam_score": final.round(1),
    })  # fmt: skip


def _listings(rng: np.random.Generator, n: int) -> pd.DataFrame:
    distance = np.exp(rng.normal(1.8, 0.6, n))
    area = np.exp(rng.normal(4.2, 0.4, n))
    age = rng.gamma(2.5, 8.0, n)
    walk = np.clip(95 - 6 * distance + rng.normal(0, 8, n), 0, 100)
    noise = 45 + 25 / (1 + distance) + rng.normal(0, 4, n)
    fee = np.clip(150 + 4.5 * area - 2 * age + rng.normal(0, 60, n), 30, None)
    price = np.exp(np.log(4.5 * area) + 0.25 - 0.045 * distance - 0.006 * age + 0.002 * walk + rng.normal(0, 0.15, n))
    days = np.exp(rng.normal(3.4 + 0.08 * (np.log(price) - 5.5) + 0.03 * distance, 0.5))
    return pd.DataFrame({
        "floor_area_m2": area.round(1), "distance_center_km": distance.round(1), "building_age_years": age.round(1),
        "walk_score": walk.round(1), "street_noise_db": noise.round(1), "condo_fee": fee.round(0),
        "listing_price_k": price.round(0), "days_on_market": days.round(0),
    })  # fmt: skip


def _air(rng: np.random.Generator, n: int) -> pd.DataFrame:
    traffic = np.exp(rng.normal(6.5, 0.6, n))
    wind = np.clip(rng.gamma(2.2, 1.5, n), 0.1, None)
    temp = rng.normal(22, 6, n)
    humidity = np.clip(rng.normal(65 - 0.8 * (temp - 22), 12), 10, 100)
    pm25 = np.exp(np.log(8) + 0.45 * np.log(traffic / 600) - 0.12 * wind + 0.004 * humidity + rng.normal(0, 0.35, n))
    pm10 = pm25 * np.exp(rng.normal(0.55, 0.15, n)) + 3
    no2 = np.exp(np.log(18) + 0.6 * np.log(traffic / 600) - 0.1 * wind + rng.normal(0, 0.3, n))
    o3 = np.clip(20 + 2.2 * (temp - 15) - 0.15 * no2 + rng.normal(0, 8, n), 1, None)
    return pd.DataFrame({
        "traffic_vehicles_h": traffic.round(0), "wind_speed_ms": wind.round(1), "temperature_c": temp.round(1),
        "humidity_pct": humidity.round(1), "pm25_ugm3": pm25.round(1), "pm10_ugm3": pm10.round(1),
        "no2_ugm3": no2.round(1), "o3_ugm3": o3.round(1),
    })  # fmt: skip


DOMAINS = (
    Domain(
        990001,
        "Vale Serra Outpatient Metabolic Survey",
        "Cross-sectional survey run in 2031 by the fictional Vale Serra regional clinic network to plan a diabetes and "
        "cardiovascular prevention program. Adults seen in routine outpatient check-ups were measured by nurses with "
        "calibrated scales and automatic blood-pressure monitors; blood was drawn after an overnight fast and analysed in "
        "the network's central laboratory, and physical activity and sleep were self-reported in a short questionnaire. "
        "Each row is one adult patient.",
        {
            "age_years": "age of the patient, in years",
            "bmi_kg_m2": "body mass index, in kg/m2",
            "activity_min_week": "self-reported moderate or vigorous physical activity, in minutes per week",
            "sleep_hours": "self-reported average sleep per night, in hours",
            "systolic_bp_mmhg": "systolic blood pressure, in mmHg",
            "fasting_glucose_mg_dl": "fasting plasma glucose, in mg/dL",
            "hba1c_pct": "glycated haemoglobin (HbA1c), a three-month average of blood glucose, in %",
            "ldl_cholesterol_mg_dl": "LDL cholesterol, in mg/dL",
        },
        "Glucose rises with BMI and with age and is a little lower in more active patients; HbA1c follows glucose "
        "closely; blood pressure rises with age and BMI; LDL cholesterol rises with age and BMI; heavier patients "
        "report less activity and slightly less sleep.",
        _clinic,
    ),
    Domain(
        990002,
        "Lumen Bay Household Energy Panel",
        "Monthly panel of homes in the fictional coastal city of Lumen Bay, collected in 2030 by the city utility to "
        "size an efficiency subsidy. Consumption comes from smart meters, the bill from the utility's billing system, "
        "home and equipment data from an enrolment interview, and temperature from the nearest weather station. Each "
        "row is one household in one summer month.",
        {
            "household_income_k": "annual household income, in thousand currency units",
            "floor_area_m2": "heated and cooled floor area of the home, in m2",
            "outdoor_temp_c": "average outdoor temperature of the month, in degrees Celsius",
            "ac_capacity_kw": "installed air-conditioning capacity, in kW",
            "appliance_age_years": "average age of the main appliances, in years",
            "solar_kwh_month": "energy generated by rooftop solar panels in the month, in kWh",
            "monthly_kwh": "electricity drawn from the grid in the month, in kWh",
            "monthly_bill": "electricity bill of the month, in currency units",
        },
        "Larger and richer homes use more energy; air conditioning drives consumption up on hot months; older "
        "appliances use more; solar generation reduces what is drawn from the grid; the bill is consumption times a "
        "tariff that varies between plans, plus a fixed fee.",
        _energy,
    ),
    Domain(
        990003,
        "Rio Claro Smallholder Yield Monitoring",
        "Field programme of the fictional Rio Claro agricultural cooperative (2029-2031) that follows maize plots of "
        "member farmers to advise on irrigation and fertilisation. Extension agents record inputs and soil tests, rain "
        "gauges on each farm record rainfall, and the harvest is weighed at the cooperative. Each row is one plot in "
        "one growing season.",
        {
            "season_rainfall_mm": "total rainfall in the growing season, in mm",
            "irrigation_hours": "hours of irrigation applied in the season",
            "fertilizer_kg_ha": "nitrogen fertiliser applied, in kg per hectare",
            "soil_ph": "soil pH before planting",
            "mean_temp_c": "mean air temperature of the season, in degrees Celsius",
            "pest_index": "pest pressure score from field scouting (higher means more pests)",
            "plot_area_ha": "plot area, in hectares",
            "yield_t_ha": "harvested grain yield, in tonnes per hectare",
        },
        "Yield increases with rainfall, irrigation and fertiliser, is highest near pH 6.5 and near 24 degrees, and "
        "drops with pest pressure; pests are worse in warmer seasons; plot size does not affect yield per hectare.",
        _farm,
    ),
    Domain(
        990004,
        "Harbor Lane Online Grocery Customers",
        "Monthly customer summary of the fictional Harbor Lane online grocery service (2031), built from the shop's "
        "web analytics, the order system and an in-app satisfaction survey, to study loyalty. Each row is one active "
        "customer in one month.",
        {
            "customer_age": "age of the customer, in years",
            "distance_km": "distance from the warehouse to the customer's address, in km",
            "sessions_per_month": "average number of shopping sessions per month",
            "avg_session_minutes": "average duration of a shopping session, in minutes",
            "discount_pct": "average discount received on orders, in %",
            "delivery_hours": "average time from order to delivery, in hours",
            "monthly_spend": "total amount spent in the month, in currency units",
            "satisfaction_score": "average satisfaction rating given in the app, from 0 to 10",
        },
        "Customers who shop more often, browse longer and get larger discounts spend more; younger customers shop more "
        "often in shorter sessions; deliveries to farther addresses take longer, and slower deliveries lower "
        "satisfaction.",
        _grocery,
    ),
    Domain(
        990005,
        "Kestrel Works Extrusion Line Log",
        "Hourly process log of a plastic-pipe extrusion line at the fictional Kestrel Works factory (2030), recorded "
        "by the line's control system and the quality lab, to plan preventive maintenance. Each row is one hour of "
        "production.",
        {
            "hours_since_maintenance": "hours since the last preventive maintenance of the line",
            "screw_speed_rpm": "extruder screw speed, in revolutions per minute",
            "ambient_humidity_pct": "relative humidity in the hall, in %",
            "barrel_temp_c": "barrel temperature, in degrees Celsius",
            "vibration_mm_s": "gearbox vibration velocity, in mm/s",
            "melt_pressure_bar": "melt pressure at the die, in bar",
            "motor_power_kw": "motor power draw, in kW",
            "defects_per_1000": "defective pieces per 1000 produced, from the quality lab",
        },
        "Vibration and barrel temperature creep up as time since maintenance grows; faster screw speeds raise "
        "temperature, pressure and power; defects increase with vibration, with temperature away from about 212 "
        "degrees and slightly with humidity.",
        _extrusion,
    ),
    Domain(
        990006,
        "Northfield District Student Outcomes Study",
        "Study of the fictional Northfield school district (2031) linking school records and a family questionnaire "
        "to understand final-year exam results. Each row is one final-year student.",
        {
            "household_income_k": "annual family income, in thousand currency units",
            "prior_year_score": "average grade in the previous school year, from 0 to 100",
            "study_hours_week": "self-reported hours of study at home per week",
            "attendance_pct": "share of school days attended, in %",
            "commute_minutes": "one-way travel time to school, in minutes",
            "sleep_hours": "self-reported average sleep per night, in hours",
            "extracurricular_hours": "hours of extracurricular activities per week",
            "final_exam_score": "score in the final exam, from 0 to 100",
        },
        "Final scores rise with the previous year's grades, study time, attendance and sleep, and fall with long "
        "commutes; better previous grades come with more study and attendance; richer families' students had "
        "slightly higher previous grades; long commutes cut sleep.",
        _school,
    ),
    Domain(
        990007,
        "Marisol Coast Apartment Listings",
        "Listings of apartments for sale gathered in 2031 by the fictional Marisol Coast real-estate observatory from "
        "agencies in the region, to track housing affordability. Each row is one apartment listing.",
        {
            "floor_area_m2": "private floor area, in m2",
            "distance_center_km": "distance to the city centre, in km",
            "building_age_years": "age of the building, in years",
            "walk_score": "walkability score of the address, from 0 to 100",
            "street_noise_db": "average daytime street noise at the facade, in dB",
            "condo_fee": "monthly condominium fee, in currency units",
            "listing_price_k": "asking price, in thousand currency units",
            "days_on_market": "days between listing and the observatory's snapshot",
        },
        "Prices grow with area and fall with distance to the centre and with building age; central addresses are more "
        "walkable and noisier; condominium fees grow with area and are lower in older buildings; more expensive and "
        "more distant apartments stay longer on the market.",
        _listings,
    ),
    Domain(
        990008,
        "Arden Valley Air Monitoring Network",
        "Hourly readings from the fictional Arden Valley urban air-quality network (2030), combining reference "
        "analysers at roadside stations, a co-located weather mast and induction-loop traffic counters, used to "
        "issue pollution alerts. Each row is one station-hour.",
        {
            "traffic_vehicles_h": "vehicles per hour on the adjacent road",
            "wind_speed_ms": "wind speed, in m/s",
            "temperature_c": "air temperature, in degrees Celsius",
            "humidity_pct": "relative humidity, in %",
            "pm25_ugm3": "fine particulate matter PM2.5, in micrograms per m3",
            "pm10_ugm3": "coarse particulate matter PM10, in micrograms per m3",
            "no2_ugm3": "nitrogen dioxide, in micrograms per m3",
            "o3_ugm3": "ozone, in micrograms per m3",
        },
        "Traffic raises PM2.5, PM10 and NO2, and wind disperses them; PM10 includes PM2.5 and tracks it closely; "
        "ozone forms on warm hours and is consumed by NO2; humidity is lower on warm hours.",
        _air,
    ),
)


def describe(domain: Domain, table: pd.DataFrame) -> str:
    """O contexto detalhado: a história, cada coluna com significado, unidade e faixa típica (do percentil 5 ao 95 dos
    dados completos) e as relações conhecidas. Nada sobre dados faltantes."""
    lines = [f"Dataset: {domain.name}.", "", domain.story, "", "Columns:"]
    for column, meaning in domain.columns.items():
        low, high = np.percentile(table[column], [5, 95])
        lines.append(f"- {column}: {meaning} (typical range {low:.3g} to {high:.3g}).")
    lines += ["", f"Known relationships: {domain.relationships}"]
    text = "\n".join(lines)
    assert not any(w in text.lower() for w in ("missing", "censor", "nonresponse", "mcar", "mnar", "imput")), text
    return text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gerar(out: Path) -> None:
    (out / "csv").mkdir(parents=True, exist_ok=False)
    sources, contexts = [], {}
    for domain in DOMAINS:
        table = domain.generate(np.random.default_rng([GENERATOR_SEED, domain.source_id]), N_ROWS).astype(float)
        assert list(table.columns) == list(domain.columns), domain.name
        path = out / "csv" / f"{domain.source_id}.csv"
        table.to_csv(path)  # com a coluna de índice, como os CSV do MechDetect
        sources.append({"openml_id": domain.source_id, "family_id": domain.source_id, "representative": True,
                        "role": "dev", "numeric_columns": ";".join(table.columns), "sha256_raw": _sha256(path),
                        "name": domain.name})  # fmt: skip
        contexts[str(domain.source_id)] = {"name": domain.name, "description": describe(domain, table)}
    pd.DataFrame(sources).to_csv(out / "sources.csv", index=False)
    (out / "contexts.json").write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "generator_seed": GENERATOR_SEED,
                "n_rows": N_ROWS, "script_sha256": _sha256(Path(__file__)),
                "files": {str(p.relative_to(out)): _sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(DOMAINS)} bases em {out}")


def _fmt(value: float) -> str:
    if np.isnan(value):
        return NA_TOKEN
    return np.format_float_positional(float(f"{value:.3g}"), trim="-")


def raw_table_csv(instance: ObservedInstance, seed: int, names: list[str]) -> str:
    """A tabela do S3: as mesmas linhas e a mesma ordem de colunas da tabela cega, com os valores originais."""
    rng = np.random.default_rng(seed)
    n = len(instance.y)
    rows = rng.choice(n, size=min(MAX_ROWS, n), replace=False)  # o mesmo sorteio de render.blind_table
    columns = [np.asarray(instance.y, dtype=float)] + [instance.z[c].to_numpy(dtype=float) for c in instance.z.columns]
    order = rng.permutation(len(columns))
    lines = [",".join(names[i] for i in order)]
    for r in rows:
        lines.append(",".join(_fmt(columns[i][r]) for i in order))
    return "\n".join(lines) + "\n"


def _scaled_named_csv(instance: ObservedInstance, seed: int, names: list[str]) -> str:
    table = blind_table(instance, seed, names)
    lines = [",".join(table.columns)]
    lines += [",".join(NA_TOKEN if np.isnan(v) else str(int(v)) for v in row) for row in table.to_numpy(float)]
    return "\n".join(lines) + "\n"


def prompts(instance: ObservedInstance, seed: int, target: str, description: str) -> dict[str, tuple[str, str]]:
    names = [target, *[str(c) for c in instance.z.columns]]
    scaled = _scaled_named_csv(instance, seed, names)
    raw = raw_table_csv(instance, seed, names)
    assert scaled.splitlines()[0] == raw.splitlines()[0]  # mesma ordem de colunas
    assert [[v == NA_TOKEN for v in line.split(",")] for line in scaled.splitlines()[1:]] == [
        [v == NA_TOKEN for v in line.split(",")] for line in raw.splitlines()[1:]
    ]  # mesmas linhas, mesmos buracos
    return {
        "S0": (A1_SYSTEM, render_a1(instance, seed)),
        "S1": (S1_SYSTEM, scaled),
        "S2": (S2_SYSTEM, f"Description: {description}\n\nTable:\n{scaled}"),
        "S3": (S3_SYSTEM, f"Description: {description}\n\nTable:\n{raw}"),
    }


def _panel_with_keys(run_dir: Path, config) -> dict[str, pd.DataFrame]:
    panel = build_panel(run_dir, config)
    panel = panel[panel.replicate == 0].reset_index(drop=True)  # sem a repetição de 10%, para economizar
    contexts = pd.read_parquet(run_dir / "contexts.parquet").set_index("context_id")
    texts = json.loads((repo_root() / "data_rb" / "contexto_sintetico" / "contexts.json").read_text(encoding="utf-8"))
    cache = ResponseCache(cache_root(run_dir, config))
    keys: dict[str, list[str]] = {arm: [] for arm in ARMS}
    by_context: dict[str, dict[str, set[str]]] = {}
    for row in panel.itertuples():
        context = contexts.loc[row.context_id]
        seed = child_seed(config.root_seed, "llm_render", row.context_id)
        description = texts[str(int(context.openml_id))]["description"]
        for arm, (system, user) in prompts(read_observed(run_dir, row.sample_id), seed, str(context.target),
                                           description).items():  # fmt: skip
            request = build_request(arm=f"ctx-{arm}", model=MODEL, system=system, user=user, schema=A1_SCHEMA,
                                    thinking_level="medium", max_output_tokens=32_768, renderer_version=RENDERER)  # fmt: skip
            keys[arm].append(cache.register(request))
            if arm in ("S2", "S3"):  # o contexto e o cabeçalho são os mesmos nos três mecanismos do trio
                head = user.split("Table:\n", 1)
                by_context.setdefault(f"{row.context_id}-{arm}", {}).setdefault("head", set()).add(
                    head[0] + head[1].splitlines()[0]
                )
    assert all(len(v["head"]) == 1 for v in by_context.values()), "contexto diferente entre os mecanismos de um trio"
    return {arm: panel.assign(key=keys[arm]) for arm in ARMS}


def perguntar(run_dir: Path, budget_only: bool) -> None:
    config = load_config(repo_root() / "configs" / "rb" / "contexto-sintetico.yaml")
    panels = _panel_with_keys(run_dir, config)
    cache = ResponseCache(cache_root(run_dir, config))
    backend = GeminiBackend(load_api_keys(repo_root() / ".env"))
    work = run_dir / "contexto-sintetico" / "_work"
    work.mkdir(parents=True, exist_ok=True)
    runner = Runner(backend, cache, MODEL, events=work / "events.jsonl")
    keys = list(dict.fromkeys(k for arm in ARMS for k in panels[arm].key))
    gate = budget_gate(runner, run_dir, config, keys, "ctx", "batch", work, None)
    print("projeção:", gate.read_text().splitlines()[-1])
    if budget_only:
        return
    runner.run_batch(keys, work / "jobs.jsonl", "rb-contexto-sintetico")
    print("respostas no caderno:", sum(cache.first_valid(k) is not None for k in keys), "de", len(keys))


def _mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    only_b, only_a = int(((~a) & b).sum()), int((a & ~b).sum())
    p = binomtest(only_b, only_b + only_a, 0.5).pvalue if only_a + only_b else 1.0
    return {"so_o_segundo_acerta": only_b, "so_o_primeiro_acerta": only_a, "p_mcnemar": float(p)}


def analisar(run_dir: Path, out: Path) -> None:
    config = load_config(repo_root() / "configs" / "rb" / "contexto-sintetico.yaml")
    panels = _panel_with_keys(run_dir, config)
    cache = ResponseCache(cache_root(run_dir, config))
    out.mkdir(parents=True, exist_ok=True)
    preds = {arm: predictions_from_cache(cache, panels[arm], arm, config.root_seed) for arm in ARMS}
    lines = ["# Contexto detalhado em bases sintéticas inventadas (exploratório; gemini-3.8-flash; 30/10/2026)", ""]
    lines.append(
        f"{len(preds['S0'])} instâncias por braço, 8 bases inventadas, 18 instâncias por base (6 de cada tipo)."
    )
    lines.append("")
    summary = []
    for arm, p in preds.items():
        proba = p[[f"p_{c}" for c in CLASSES]].to_numpy()
        m = metrics(p.y_true.to_numpy(), proba, p.y_pred.to_numpy())
        per_family, _ = summarize_by_family(p)
        summary.append({"braço": arm, "falhas": int((p.status == "failed").sum()),
                        "macro_recall": m["macro_recall"], "recall_MCAR": m["recall_MCAR"], "recall_MAR": m["recall_MAR"],
                        "recall_MNAR": m["recall_MNAR"], "log_loss": m["log_loss"],
                        "macro_recall_por_base_media": per_family.macro_recall.mean()})  # fmt: skip
        p.to_parquet(out / f"predicoes_{arm}.parquet", index=False)
    table = pd.DataFrame(summary)
    lines += ["## Acerto por braço", table.round(3).to_string(index=False), ""]
    correct = {arm: (p.sort_values("sample_id").y_pred.to_numpy() == p.sort_values("sample_id").y_true.to_numpy())
               for arm, p in preds.items()}  # fmt: skip
    family = preds["S0"].sort_values("sample_id").family_id.to_numpy()
    pairs = []
    for first, second in (("S0", "S1"), ("S0", "S2"), ("S0", "S3"), ("S1", "S2"), ("S2", "S3")):
        per = pd.DataFrame({"f": family, "a": correct[first], "b": correct[second]}).groupby("f")[["a", "b"]].mean()
        d = (per.b - per.a).to_numpy()
        low, high = bootstrap_hl_ci(d)
        pairs.append({"comparação": f"{second} − {first}", "dif_acerto": float(correct[second].mean() - correct[first].mean()),
                      "HL_por_base": hodges_lehmann(d), "ic_baixo": low, "ic_alto": high,
                      **_mcnemar(correct[first], correct[second])})  # fmt: skip
    lines += ["## Comparações pareadas (mesmas instâncias)", pd.DataFrame(pairs).round(3).to_string(index=False), ""]
    for arm, p in preds.items():
        cm = pd.crosstab(
            p.y_true.map(dict(enumerate(CLASSES))), p.y_pred.map({-1: "falha", **dict(enumerate(CLASSES))})
        )
        lines += [f"## Matriz de confusão, {arm}", cm.to_string(), ""]
    usage = []
    for arm in ARMS:
        attempts = [a for k in panels[arm].key for a in cache.attempts(k)]
        usage.append({"braço": arm, "chamadas": len(attempts),
                      "entrada_média": np.mean([a["usage"].get("prompt_token_count") or 0 for a in attempts]),
                      "raciocínio_médio": np.mean([a["usage"].get("thoughts_token_count") or 0 for a in attempts])})  # fmt: skip
    lines += ["## Tokens", pd.DataFrame(usage).round(0).to_string(index=False)]
    (out / "relatorio.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(out / "resumo.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "comparacoes.csv", index=False)
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gerar")
    p.add_argument("out", type=Path)
    p = sub.add_parser("perguntar")
    p.add_argument("run", type=Path)
    p.add_argument("--budget-only", action="store_true")
    p = sub.add_parser("analisar")
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "gerar":
        gerar(args.out)
    elif args.command == "perguntar":
        perguntar(args.run, args.budget_only)
    else:
        analisar(args.run, args.out)


if __name__ == "__main__":
    main()
