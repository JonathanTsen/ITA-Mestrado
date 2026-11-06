"""As 30 bases inventadas da H7 (adendo LLM, §10; inclusão decidida pelo autor em 03/11/2026).

Domínios novos, escritos em 03/11/2026 sem nenhuma chamada ao LLM e sem olhar resultados, no mesmo molde dos 16 de
desenvolvimento de `contexto_sintetico.py` (`DOMAINS` e `DOMAINS_2`), com temas diferentes dos deles: 1.000 linhas e 8
colunas numéricas, uma história de coleta, o significado e a unidade de cada coluna e as relações conhecidas. O
contexto nunca fala de dados faltantes (a checagem fica em `contexto_sintetico.describe`). IDs 990201–990230.
"""

import numpy as np
import pandas as pd
from contexto_sintetico import Domain


def _dairy(rng: np.random.Generator, n: int) -> pd.DataFrame:
    parity = rng.poisson(1.6, n) + 1.0
    dim = rng.uniform(5, 305, n)
    age = 22 + 13 * (parity - 1) + dim / 30 + rng.uniform(0, 4, n)
    weight = np.clip(rng.normal(560 + 25 * np.minimum(parity, 4), 55), 380, 820)
    milk = np.clip(
        14 + 4.5 * np.minimum(parity, 3) + 22 * (dim / 60) * np.exp(1 - dim / 60) + 0.01 * (weight - 560)
        + rng.normal(0, 4.5, n), 3, None,
    )  # fmt: skip
    fat = np.clip(4.6 - 0.035 * milk + 0.002 * dim + rng.normal(0, 0.3, n), 2.5, 6.5)
    protein = np.clip(0.55 + 0.62 * fat + rng.normal(0, 0.15, n), 2.4, 4.8)
    scc = np.exp(rng.normal(4.4 + 0.18 * parity + 0.0015 * dim, 0.8))
    feed = np.clip(10 + 0.35 * milk + 0.01 * (weight - 560) + rng.normal(0, 1.6, n), 8, 35)
    return pd.DataFrame({
        "cow_age_months": age.round(1), "days_in_milk": dim.round(0), "body_weight_kg": weight.round(0),
        "milk_kg_day": milk.round(1), "fat_pct": fat.round(2), "protein_pct": protein.round(2),
        "somatic_cells_k_ml": scc.round(0), "dry_matter_intake_kg": feed.round(1),
    })  # fmt: skip


def _turbine(rng: np.random.Generator, n: int) -> pd.DataFrame:
    wind = np.clip(rng.weibull(2.1, n) * 8.5, 0.3, 25)
    temp = rng.normal(11, 7, n)
    density = 1.225 * (288 / (273 + temp)) + rng.normal(0, 0.01, n)
    power = np.clip(2300 / (1 + np.exp(-(wind - 9.0) / 1.3)) * density / 1.225 + rng.normal(0, 60, n), 0, 2400)
    rotor = np.clip(6 + 1.1 * np.minimum(wind, 12) + rng.normal(0, 0.6, n), 0, 16)
    pitch = np.clip(np.where(wind > 12, 2.2 * (wind - 12), 0) + rng.normal(0.5, 0.6, n), -1, 30)
    gearbox = 38 + 0.012 * power + 0.6 * temp + rng.normal(0, 2.5, n)
    vibration = np.clip(0.8 + 0.18 * rotor + 0.0004 * power + np.exp(rng.normal(-1.5, 0.6, n)), 0.3, None)
    return pd.DataFrame({
        "wind_speed_m_s": wind.round(2), "ambient_temp_c": temp.round(1), "air_density_kg_m3": density.round(3),
        "active_power_kw": power.round(0), "rotor_speed_rpm": rotor.round(2), "blade_pitch_deg": pitch.round(1),
        "gearbox_oil_temp_c": gearbox.round(1), "nacelle_vibration_mm_s": vibration.round(2),
    })  # fmt: skip


def _freight(rng: np.random.Generator, n: int) -> pd.DataFrame:
    distance = np.exp(rng.normal(5.6, 0.7, n))
    load = np.clip(rng.normal(17, 6, n), 1, 30)
    driver_years = rng.gamma(2.5, 4.0, n)
    hours = distance / np.clip(rng.normal(68, 7, n), 40, 90) + rng.gamma(1.5, 0.8, n)
    fuel = np.clip(distance * (0.24 + 0.006 * load) * rng.normal(1, 0.06, n), 5, None)
    unloading = np.clip(rng.gamma(2, 15, n) + 0.08 * distance + 1.5 * load, 5, None)
    delay = np.clip(rng.normal(-10 + 0.5 * unloading - 1.0 * driver_years, 30), -30, None)
    revenue = 150 + 1.6 * distance + 12 * load + rng.normal(0, 120, n)
    return pd.DataFrame({
        "trip_distance_km": distance.round(0), "load_tonnes": load.round(1), "driver_experience_years": driver_years.round(1),
        "driving_hours": hours.round(2), "fuel_litres": fuel.round(0), "unloading_minutes": unloading.round(0),
        "arrival_delay_min": delay.round(0), "trip_revenue": revenue.round(0),
    })  # fmt: skip


def _hotel(rng: np.random.Generator, n: int) -> pd.DataFrame:
    lead = np.exp(rng.normal(3.3, 1.0, n))
    home = np.exp(rng.normal(5.5, 1.1, n))
    room = np.clip(rng.normal(28, 7, n), 14, 70)
    rate = np.clip(60 + 2.2 * room - 0.15 * lead + rng.normal(0, 22, n), 40, None)
    stay = np.clip(rng.gamma(2.2, 18, n) + 0.02 * home, 10, None)
    spend = np.clip(stay / 24 * (8 + 0.12 * rate) * np.exp(rng.normal(0, 0.5, n)), 0, None)
    score = np.clip(7.2 + 0.04 * (room - 28) - 0.004 * (rate - 120) + rng.normal(0, 1.0, n), 1, 10)
    age = np.clip(rng.normal(44, 13, n), 18, 85)
    return pd.DataFrame({
        "booking_lead_days": lead.round(0), "home_distance_km": home.round(0), "room_size_m2": room.round(1),
        "nightly_rate": rate.round(0), "extras_spend": spend.round(1), "stay_hours": stay.round(1),
        "review_score": score.round(1), "guest_age_years": age.round(0),
    })  # fmt: skip


def _hive(rng: np.random.Generator, n: int) -> pd.DataFrame:
    temp_out = rng.normal(18, 7, n)
    humidity = np.clip(rng.normal(65 - 0.8 * (temp_out - 18), 12), 20, 100)
    colony = np.clip(rng.normal(28 + 0.4 * temp_out, 8), 4, 60)
    temp_in = np.clip(34.5 - 0.02 * np.abs(temp_out - 25) * (40 - colony) / 10 + rng.normal(0, 0.7, n), 20, 37)
    traffic = np.clip(12 * colony * np.clip((temp_out - 8) / 15, 0, 1.4) + rng.normal(0, 60, n), 0, None)
    weight = np.clip(18 + 0.6 * colony + rng.normal(0, 4, n), 8, None)
    gain = 0.004 * traffic - 0.15 + rng.normal(0, 0.3, n)
    varroa = np.exp(rng.normal(0.6 + 0.02 * colony, 0.7))
    return pd.DataFrame({
        "outside_temp_c": temp_out.round(1), "outside_humidity_pct": humidity.round(0),
        "colony_frames_k_bees": colony.round(1), "brood_temp_c": temp_in.round(1), "forager_trips_hour": traffic.round(0),
        "hive_weight_kg": weight.round(1), "daily_weight_gain_kg": gain.round(2), "mites_per_100_bees": varroa.round(1),
    })  # fmt: skip


def _ski(rng: np.random.Generator, n: int) -> pd.DataFrame:
    snowfall = np.exp(rng.normal(1.5, 1.1, n))
    base = np.clip(rng.normal(120, 40, n) + 1.5 * snowfall, 20, None)
    temp = rng.normal(-5, 5, n)
    occupancy = np.clip(rng.normal(62, 18, n), 10, 100)
    visitors = np.clip(600 + 40 * occupancy + 25 * snowfall - 40 * np.abs(temp + 4) + rng.normal(0, 450, n), 150, None)
    queue = np.clip(2 + 0.004 * visitors + rng.normal(0, 2.5, n), 0, None)
    lift_hours = np.clip(rng.normal(110, 15, n) - 0.4 * np.maximum(snowfall - 30, 0), 30, None)
    revenue = visitors * np.clip(rng.normal(58, 6, n), 35, None) / 1000
    return pd.DataFrame({
        "snowfall_cm": snowfall.round(0), "snow_base_cm": base.round(0), "summit_temp_c": temp.round(1),
        "valley_hotel_occupancy_pct": occupancy.round(1), "skier_visits": visitors.round(0),
        "mean_lift_queue_min": queue.round(1), "lift_hours_operated": lift_hours.round(1),
        "ticket_revenue_k": revenue.round(1),
    })  # fmt: skip


def _auction(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = rng.uniform(1, 16, n)
    mileage = np.clip(age * rng.normal(15, 4, n), 1, None)
    engine = np.clip(rng.normal(1.9, 0.5, n), 1.0, 4.5)
    condition = np.clip(rng.normal(4.0 - 0.12 * age, 0.6), 1, 5)
    watchers = rng.poisson(40 + 15 * condition, n).astype(float)
    reserve = np.clip(26000 * np.exp(-0.11 * age) * (1 + 0.15 * (engine - 1.9)) * rng.normal(1, 0.08, n), 900, None)
    price = np.clip(reserve * (0.85 + 0.003 * watchers + 0.05 * (condition - 3)) * rng.normal(1, 0.07, n), 500, None)
    days = np.clip(rng.gamma(2, 7, n) - 0.12 * watchers + 6, 0, None)
    return pd.DataFrame({
        "vehicle_age_years": age.round(1), "odometer_k_km": mileage.round(0), "engine_litres": engine.round(2),
        "condition_grade": condition.round(1), "online_watchers": watchers, "reserve_price": reserve.round(-1),
        "hammer_price": price.round(-1), "days_on_lot": days.round(1),
    })  # fmt: skip


def _forest(rng: np.random.Generator, n: int) -> pd.DataFrame:
    elevation = rng.uniform(200, 1800, n)
    slope = np.clip(rng.gamma(2.5, 7, n), 0, 60)
    age = np.clip(rng.normal(55, 25, n), 8, 160)
    dbh = np.clip(6 + 0.45 * age - 0.004 * (elevation - 800) + rng.normal(0, 6, n), 5, None)
    height = np.clip(1.3 + 30 * (1 - np.exp(-0.045 * dbh)) - 0.002 * (elevation - 800) + rng.normal(0, 2, n), 2, None)
    density = np.clip(1600 * np.exp(-0.012 * age) + rng.normal(0, 120, n), 80, None)
    basal = density * np.pi * (dbh / 200) ** 2 * rng.normal(1, 0.1, n)
    litter = np.clip(rng.normal(4 + 0.03 * age - 0.04 * slope, 1.5), 0.2, None)
    return pd.DataFrame({
        "elevation_m": elevation.round(0), "slope_deg": slope.round(1), "stand_age_years": age.round(0),
        "mean_dbh_cm": dbh.round(1), "dominant_height_m": height.round(1), "trees_per_ha": density.round(0),
        "basal_area_m2_ha": basal.round(1), "litter_depth_cm": litter.round(1),
    })  # fmt: skip


def _river(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rain = np.exp(rng.normal(2.0, 1.2, n))
    area = np.exp(rng.normal(6.0, 1.0, n))
    stage = np.clip(0.4 + 0.25 * np.log1p(area) / 6 * np.log1p(rain) + rng.normal(0, 0.15, n), 0.1, None)
    flow = np.clip(4.5 * stage**1.7 * (area / 400) ** 0.6 * rng.normal(1, 0.1, n), 0.05, None)
    turbidity = np.exp(rng.normal(1.5 + 0.35 * np.log1p(rain), 0.6))
    water_temp = rng.normal(14, 4, n)
    oxygen = np.clip(12.5 - 0.22 * water_temp - 0.01 * turbidity + rng.normal(0, 0.6, n), 3, 15)
    conductivity = np.exp(rng.normal(5.5 - 0.15 * np.log1p(rain), 0.35))
    return pd.DataFrame({
        "rain_24h_mm": rain.round(1), "catchment_km2": area.round(0), "stage_m": stage.round(2),
        "discharge_m3_s": flow.round(2), "turbidity_ntu": turbidity.round(1), "water_temp_c": water_temp.round(1),
        "dissolved_oxygen_mg_l": oxygen.round(2), "conductivity_us_cm": conductivity.round(0),
    })  # fmt: skip


def _datacenter(rng: np.random.Generator, n: int) -> pd.DataFrame:
    servers = rng.integers(8, 42, n).astype(float)
    utilization = np.clip(rng.beta(3, 3, n) * 100, 2, 99)
    inlet = np.clip(rng.normal(22, 1.8, n), 16, 30)
    power = servers * (0.18 + 0.0035 * utilization) * rng.normal(1, 0.05, n)
    outlet = inlet + 2.2 * power / np.clip(rng.normal(1.6, 0.25, n), 0.8, None)
    fans = np.clip(30 + 2.0 * (outlet - 30) + 0.2 * utilization + rng.normal(0, 5, n), 15, 100)
    errors = rng.poisson(np.exp(-3 + 0.12 * (outlet - 30)) * servers, n).astype(float)
    pue = np.clip(1.18 + 0.01 * (inlet - 22) + rng.normal(0, 0.04, n), 1.05, 2)
    return pd.DataFrame({
        "servers_in_rack": servers, "cpu_utilization_pct": utilization.round(1), "inlet_temp_c": inlet.round(1),
        "rack_power_kw": power.round(2), "outlet_temp_c": outlet.round(1), "fan_speed_pct": fans.round(0),
        "corrected_memory_errors": errors, "site_pue": pue.round(3),
    })  # fmt: skip


def _airport(rng: np.random.Generator, n: int) -> pd.DataFrame:
    seats = rng.choice([76.0, 120, 150, 180, 220, 300], n, p=[0.15, 0.15, 0.3, 0.2, 0.1, 0.1])
    load = np.clip(rng.beta(8, 2, n) * 100, 30, 100)
    passengers = np.round(seats * load / 100)
    bags = np.clip(passengers * rng.normal(0.85, 0.15, n), 0, None)
    inbound_delay = np.clip(rng.gamma(1.2, 12, n) - 5, -15, None)
    crew = np.clip(rng.normal(4 + seats / 60, 0.8), 2, 12)
    turnaround = np.clip(25 + 0.08 * seats + 0.03 * bags - 1.2 * crew + rng.normal(0, 6, n), 18, None)
    departure_delay = np.clip(0.6 * inbound_delay + 0.5 * (turnaround - 40) + rng.normal(0, 8, n), -10, None)
    fuel = np.clip(seats * rng.normal(45, 8, n), 1500, None)
    return pd.DataFrame({
        "passengers": passengers, "load_factor_pct": load.round(1), "checked_bags": bags.round(0),
        "inbound_delay_min": inbound_delay.round(0), "ground_crew_hours": (crew * turnaround / 60).round(2),
        "turnaround_min": turnaround.round(0), "departure_delay_min": departure_delay.round(0),
        "fuel_uplift_kg": fuel.round(-1),
    })  # fmt: skip


def _roastery(rng: np.random.Generator, n: int) -> pd.DataFrame:
    green = np.clip(rng.normal(12, 3, n), 4, 25)
    moisture = np.clip(rng.normal(10.8, 0.9, n), 8, 13.5)
    charge = np.clip(rng.normal(200, 10, n), 170, 235)
    roast = np.clip(9 + 0.15 * green + 0.4 * (moisture - 10.8) - 0.03 * (charge - 200) + rng.normal(0, 0.8, n), 7, 18)
    end_temp = np.clip(205 + 1.3 * roast + rng.normal(0, 3, n), 190, 240)
    loss = np.clip(11 + 0.2 * (end_temp - 210) + 0.4 * (moisture - 10.8) + rng.normal(0, 0.8, n), 8, 22)
    color = np.clip(95 - 0.9 * (end_temp - 200) + rng.normal(0, 3, n), 30, 110)
    cupping = np.clip(84 - 0.02 * (color - 65) ** 2 / 10 + rng.normal(0, 1.2, n), 75, 92)
    return pd.DataFrame({
        "green_batch_kg": green.round(1), "bean_moisture_pct": moisture.round(1), "charge_temp_c": charge.round(0),
        "roast_minutes": roast.round(1), "drop_temp_c": end_temp.round(1), "weight_loss_pct": loss.round(1),
        "agtron_color": color.round(1), "cupping_score": cupping.round(1),
    })  # fmt: skip


def _buoy(rng: np.random.Generator, n: int) -> pd.DataFrame:
    wind = np.clip(rng.weibull(2, n) * 9, 0.2, 30)
    fetch = np.exp(rng.normal(5, 0.8, n))
    height = np.clip(0.02 * wind**1.6 * (fetch / 150) ** 0.25 + rng.gamma(1.5, 0.3, n), 0.1, None)
    period = np.clip(3 + 1.6 * np.sqrt(height) * 3 + rng.normal(0, 0.8, n), 2, 20)
    pressure = rng.normal(1013, 9, n) - 0.4 * wind
    sst = rng.normal(17, 3, n)
    air = sst + rng.normal(-0.8, 1.5, n)
    current = np.clip(0.03 * wind + rng.gamma(2, 0.08, n), 0, None)
    return pd.DataFrame({
        "wind_speed_m_s": wind.round(1), "fetch_km": fetch.round(0), "significant_wave_m": height.round(2),
        "peak_period_s": period.round(1), "sea_level_pressure_hpa": pressure.round(1),
        "sea_surface_temp_c": sst.round(1), "air_temp_c": air.round(1), "surface_current_m_s": current.round(2),
    })  # fmt: skip


def _gym(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = np.clip(rng.normal(36, 12, n), 16, 80)
    tenure = rng.gamma(1.6, 10, n)
    minutes = np.clip(rng.normal(62 - 0.2 * (age - 36), 15), 15, 180)
    hours = np.clip(rng.gamma(3, 3.3, n) * np.clip(1 - 0.006 * (age - 36) + 0.006 * tenure, 0.3, None), 0, None)
    class_share = np.clip(rng.beta(2, 4, n) * 100, 0, 100)
    fee = np.clip(rng.normal(45, 9, n) - 0.2 * tenure, 20, None)
    trainer = np.clip(np.exp(rng.normal(2.5 + 0.03 * fee, 1.0)) - 20, 0, None)
    rating = np.clip(6.5 + 0.12 * hours - 0.02 * (fee - 45) + rng.normal(0, 1.4, n), 0, 10)
    return pd.DataFrame({
        "member_age_years": age.round(0), "tenure_months": tenure.round(1), "gym_hours_month": hours.round(1),
        "mean_session_min": minutes.round(0), "group_class_share_pct": class_share.round(1),
        "monthly_fee": fee.round(2), "personal_training_spend": trainer.round(0), "satisfaction_0_10": rating.round(1),
    })  # fmt: skip


def _claims(rng: np.random.Generator, n: int) -> pd.DataFrame:
    driver_age = np.clip(rng.normal(44, 14, n), 18, 85)
    car_value = np.exp(rng.normal(9.8, 0.5, n))
    speed = np.clip(rng.gamma(3, 12, n), 3, 140)
    damage = np.clip(car_value * np.clip(0.004 * speed + rng.normal(0, 0.05, n), 0.01, 0.95), 100, None)
    injury = np.clip(np.exp(rng.normal(-2 + 0.03 * speed, 1.2)) * 1000, 0, None)
    days_report = np.clip(rng.gamma(1.4, 3.5, n) + 0.04 * (driver_age - 44), 0, None)
    adjust_days = np.clip(10 + 0.0004 * (damage + injury) + 0.6 * days_report + rng.normal(0, 6, n), 1, None)
    paid = np.clip(0.88 * (damage + injury) - 500 + rng.normal(0, 600, n), 0, None)
    return pd.DataFrame({
        "driver_age_years": driver_age.round(0), "vehicle_value": car_value.round(-1), "impact_speed_kmh": speed.round(0),
        "vehicle_damage": damage.round(0), "medical_costs": injury.round(0), "days_to_report": days_report.round(1),
        "days_to_settle": adjust_days.round(0), "amount_paid": paid.round(0),
    })  # fmt: skip


def _mortgage(rng: np.random.Generator, n: int) -> pd.DataFrame:
    income = np.exp(rng.normal(11.1, 0.45, n))
    score = np.clip(rng.normal(700 + 0.0006 * (income - 70000), 55), 450, 850)
    price = np.clip(income * rng.normal(4.2, 1.0, n), 50000, None)
    down = np.clip(rng.beta(2, 6, n) * 60 + 0.03 * (score - 700), 3, 80)
    loan = price * (1 - down / 100)
    dti = np.clip(28 + 100 * (loan * 0.065 / 12) / (income / 12) * 0.3 + rng.normal(0, 6, n), 5, 65)
    rate = np.clip(7.8 - 0.008 * (score - 700) + 0.02 * (dti - 35) + rng.normal(0, 0.3, n), 4, 12)
    tenure = np.clip(rng.gamma(2, 3.2, n), 0, 40)
    return pd.DataFrame({
        "annual_income": income.round(-2), "credit_score": score.round(0), "home_price": price.round(-3),
        "down_payment_pct": down.round(1), "loan_amount": loan.round(-3), "debt_to_income_pct": dti.round(1),
        "offered_rate_pct": rate.round(2), "job_tenure_years": tenure.round(1),
    })  # fmt: skip


def _concert(rng: np.random.Generator, n: int) -> pd.DataFrame:
    capacity = np.clip(np.exp(rng.normal(7.6, 0.8, n)), 300, 20000)
    followers = np.exp(rng.normal(11.5, 1.2, n))
    price = np.clip(18 + 6 * np.log(followers / 1e5 + 1) * 5 + rng.normal(0, 8, n), 10, None)
    sold = np.clip(
        capacity * np.clip(0.35 + 0.08 * np.log(followers / 2e4) + rng.normal(0, 0.12, n), 0.05, 1), 20, None
    )
    days = np.clip(rng.gamma(2, 25, n), 3, None)
    bar = sold * np.clip(rng.normal(14, 4, n), 2, None) / 1000
    merch = sold * np.clip(rng.normal(6, 3, n), 0, None) / 1000
    noise = np.clip(rng.normal(98 + 0.0002 * capacity, 3), 85, 115)
    return pd.DataFrame({
        "venue_capacity": capacity.round(-1), "artist_followers": followers.round(-2), "ticket_price": price.round(1),
        "tickets_sold": sold.round(0), "days_on_sale": days.round(0), "bar_sales_k": bar.round(2),
        "merch_sales_k": merch.round(2), "peak_sound_db": noise.round(1),
    })  # fmt: skip


def _donor(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = np.clip(rng.normal(41, 13, n), 18, 75)
    weight = np.clip(rng.normal(74, 13, n), 50, 140)
    hemoglobin = np.clip(rng.normal(14.2, 1.2, n), 11.5, 18.5)
    years_donor = np.clip(rng.uniform(0, 1, n) * (age - 18), 0, None)
    pulse = np.clip(rng.normal(74 - 0.25 * years_donor, 9), 48, 110)
    sbp = np.clip(rng.normal(118 + 0.4 * (age - 41) + 0.2 * (weight - 74), 12), 90, 180)
    volume = np.clip(450 + rng.normal(0, 12, n), 400, 500)
    minutes = np.clip(rng.normal(9 - 0.4 * (hemoglobin - 14.2) + 0.03 * (pulse - 74), 1.8), 4, 20)
    return pd.DataFrame({
        "donor_age_years": age.round(0), "body_weight_kg": weight.round(1), "hemoglobin_g_dl": hemoglobin.round(1),
        "years_since_first_donation": years_donor.round(1), "pulse_bpm": pulse.round(0),
        "systolic_bp_mmhg": sbp.round(0), "collected_ml": volume.round(0), "collection_minutes": minutes.round(1),
    })  # fmt: skip


def _brewery(rng: np.random.Generator, n: int) -> pd.DataFrame:
    og = np.clip(rng.normal(1.055, 0.01, n), 1.032, 1.095)
    pitch = np.clip(rng.normal(0.9, 0.25, n), 0.3, 2)
    temp = np.clip(rng.normal(18.5, 1.5, n), 14, 24)
    days = np.clip(12 - 0.3 * (temp - 18.5) - 1.5 * (pitch - 0.9) + 120 * (og - 1.055) + rng.normal(0, 1.2, n), 5, None)
    fg = np.clip(1.010 + 0.18 * (og - 1.055) - 0.0007 * (temp - 18.5) + rng.normal(0, 0.0015, n), 1.002, 1.025)
    abv = (og - fg) * 131.25
    diacetyl = np.exp(rng.normal(-3 + 0.12 * (temp - 18.5) - 0.06 * days, 0.4))
    ibu = np.clip(rng.normal(35, 12, n), 8, 90)
    return pd.DataFrame({
        "original_gravity": og.round(4), "pitch_rate_m_cells_ml": pitch.round(2), "ferment_temp_c": temp.round(1),
        "days_to_terminal": days.round(1), "final_gravity": fg.round(4), "abv_pct": abv.round(2),
        "diacetyl_mg_l": diacetyl.round(3), "bitterness_ibu": ibu.round(0),
    })  # fmt: skip


def _haul(rng: np.random.Generator, n: int) -> pd.DataFrame:
    payload = np.clip(rng.normal(210, 25, n), 120, 260)
    grade = np.clip(rng.normal(8, 2, n), 2, 14)
    distance = np.clip(rng.normal(3.2, 1.0, n), 0.8, 7)
    cycle = 6 + 4.5 * distance * (1 + 0.04 * grade) + 0.015 * payload + rng.normal(0, 2, n)
    fuel = np.clip(0.9 * distance * (1 + 0.08 * grade) * payload / 100 + rng.normal(0, 1.2, n), 1, None)
    tyre = np.clip(55 + 0.08 * payload + 0.9 * distance + rng.normal(0, 4, n), 35, 110)
    queue = np.clip(rng.gamma(1.5, 2.5, n), 0, None)
    speed = np.clip(2 * distance / ((cycle - queue * 0.3) / 60) * rng.normal(1, 0.05, n), 5, 60)
    return pd.DataFrame({
        "payload_tonnes": payload.round(0), "ramp_grade_pct": grade.round(1), "haul_distance_km": distance.round(2),
        "cycle_time_min": cycle.round(1), "fuel_litres": fuel.round(1), "tyre_temp_c": tyre.round(1),
        "shovel_queue_min": queue.round(1), "mean_speed_kmh": speed.round(1),
    })  # fmt: skip


def _port(rng: np.random.Generator, n: int) -> pd.DataFrame:
    teu = np.clip(rng.normal(9000, 4000, n), 1000, 24000)
    moves = np.clip(teu * rng.normal(0.22, 0.06, n), 150, None)
    cranes = np.clip(np.round(2 + moves / 900 + rng.normal(0, 0.6, n)), 1, 8)
    productivity = np.clip(rng.normal(28, 4, n) - 0.5 * (cranes - 4), 12, 45)
    crane_hours = moves / productivity
    berth = crane_hours / cranes + rng.gamma(1.5, 1.5, n)
    wait = np.clip(rng.gamma(1.3, 8, n) + 0.0006 * teu, 0, None)
    wind = np.clip(rng.weibull(2, n) * 8, 0, 30)
    reefers = np.clip(moves * rng.beta(2, 12, n), 0, None)
    return pd.DataFrame({
        "ship_capacity_teu": teu.round(-1), "container_moves": moves.round(0), "crane_hours": crane_hours.round(1),
        "moves_per_crane_hour": productivity.round(1), "berth_hours": berth.round(1), "anchorage_wait_hours": wait.round(1),
        "wind_gust_m_s": wind.round(1), "reefer_moves": reefers.round(0),
    })  # fmt: skip


def _gamers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    level = np.clip(rng.gamma(2, 15, n), 1, 200)
    days_since_install = np.clip(level * rng.normal(2.5, 0.8, n), 1, None)
    session = np.clip(np.exp(rng.normal(3.3, 0.6, n)), 2, 300)
    playtime = np.clip((4 + 0.03 * level) * session / 60 * np.exp(rng.normal(0, 0.4, n)), 0, None)
    friends = rng.poisson(5 + 0.5 * level, n).astype(float)
    spend = np.clip(np.exp(rng.normal(-1 + 0.02 * level + 0.008 * friends, 1.3)) - 0.3, 0, None)
    win_rate = np.clip(rng.normal(45 + 0.06 * level, 8), 5, 95)
    ping = np.exp(rng.normal(3.7, 0.4, n))
    return pd.DataFrame({
        "player_level": level.round(0), "days_since_install": days_since_install.round(0),
        "mean_session_min": session.round(1), "playtime_hours_week": playtime.round(2), "friend_list_size": friends,
        "spend_last_30d": spend.round(2), "win_rate_pct": win_rate.round(1), "median_ping_ms": ping.round(0),
    })  # fmt: skip


def _commuter(rng: np.random.Generator, n: int) -> pd.DataFrame:
    distance = np.clip(rng.normal(28, 9, n), 6, 60)
    spacing = np.clip(rng.normal(3, 0.7, n), 1.2, 6)
    stops = distance / spacing
    riders = np.clip(rng.normal(600, 220, n), 60, 1400)
    temp = rng.normal(12, 9, n)
    dwell = np.clip(25 + 0.03 * riders + rng.normal(0, 8, n), 10, None)
    delay = np.clip(
        rng.gamma(1.2, 2, n) + 0.002 * riders + 0.01 * stops * (dwell - 30) + 0.08 * np.abs(temp - 12) - 2, -2, None
    )  # fmt: skip
    crowding = np.clip(100 * riders / 900 + rng.normal(0, 8, n), 5, 180)
    energy = np.clip(distance * (14 + 0.006 * riders) + 2 * stops + rng.normal(0, 25, n), 50, None)
    return pd.DataFrame({
        "route_km": distance.round(1), "mean_stop_spacing_km": spacing.round(2), "boardings": riders.round(0),
        "outside_temp_c": temp.round(1), "mean_dwell_s": dwell.round(0), "arrival_delay_min": delay.round(1),
        "peak_load_pct": crowding.round(0), "traction_kwh": energy.round(0),
    })  # fmt: skip


def _reef(rng: np.random.Generator, n: int) -> pd.DataFrame:
    depth = np.clip(rng.gamma(4, 2.5, n), 1, 30)
    sst = rng.normal(28, 1, n)
    turbidity = np.exp(rng.normal(0.5, 0.5, n))
    coral = np.clip(45 - 0.8 * np.abs(depth - 8) - 6 * (sst - 28) - 4 * turbidity + rng.normal(0, 8, n), 1, 95)
    algae = np.clip(30 - 0.4 * coral + 3 * turbidity + rng.normal(0, 7, n), 0, 90)
    fish = np.clip(rng.poisson(np.clip(20 + 0.6 * coral, 1, None), n).astype(float), 0, None)
    biomass = np.clip(fish * rng.normal(1.8, 0.5, n) * 10, 0, None)
    urchins = rng.gamma(2, np.clip(1 + 0.04 * algae, 0.1, None) / 2)
    return pd.DataFrame({
        "transect_depth_m": depth.round(1), "water_temp_c": sst.round(2), "secchi_turbidity_index": turbidity.round(2),
        "live_coral_cover_pct": coral.round(1), "macroalgae_cover_pct": algae.round(1), "fish_species_count": fish,
        "fish_biomass_kg_ha": biomass.round(0), "urchins_per_m2": urchins.round(2),
    })  # fmt: skip


def _banding(rng: np.random.Generator, n: int) -> pd.DataFrame:
    wing = np.clip(rng.normal(72, 6, n), 50, 95)
    mass = np.clip(0.28 * wing - 6 + rng.normal(0, 1.5, n), 6, 30)
    fat = np.clip(0.12 * mass + rng.normal(0, 0.6, n), 0, None)
    tarsus = np.clip(0.27 * wing + rng.normal(0, 1, n), 12, 30)
    hour = np.clip(rng.normal(8.5, 2, n), 5, 16)
    temp = rng.normal(9 + 0.6 * (hour - 8), 4, n)
    day = rng.uniform(240, 300, n)
    stopover = np.clip(rng.gamma(2, 2, n) - 1.5 * fat + 2, 0, None)
    return pd.DataFrame({
        "wing_chord_mm": wing.round(1), "body_mass_g": mass.round(2), "fat_reserve_g": fat.round(2),
        "tarsus_mm": tarsus.round(1), "capture_hour": hour.round(2), "air_temp_c": temp.round(1),
        "day_of_year": day.round(0), "stopover_days": stopover.round(1),
    })  # fmt: skip


def _glacier(rng: np.random.Generator, n: int) -> pd.DataFrame:
    elevation = rng.uniform(2600, 3900, n)
    aspect_north = rng.uniform(-1, 1, n)
    summer_temp = 6 - 0.0065 * (elevation - 3000) + rng.normal(0, 0.6, n)
    winter_snow = np.clip(1.2 + 0.0009 * (elevation - 3000) + rng.normal(0, 0.35, n), 0.1, None)
    debris = np.clip(rng.gamma(1.2, 6, n) - 0.004 * (elevation - 3000), 0, None)
    melt = np.clip(0.75 * np.clip(summer_temp, 0, None) * (1 - 0.15 * aspect_north) * np.exp(-0.03 * debris)
                   + rng.normal(0, 0.25, n), 0, None)  # fmt: skip
    balance = winter_snow - melt + rng.normal(0, 0.15, n)
    velocity = np.clip(rng.normal(35 - 0.01 * (elevation - 3000), 10), 1, None)
    return pd.DataFrame({
        "stake_elevation_m": elevation.round(0), "aspect_north_index": aspect_north.round(2),
        "summer_mean_temp_c": summer_temp.round(2), "winter_snow_m_we": winter_snow.round(2),
        "debris_cover_cm": debris.round(1), "summer_melt_m_we": melt.round(2), "annual_balance_m_we": balance.round(2),
        "surface_velocity_m_yr": velocity.round(1),
    })  # fmt: skip


def _elevator(rng: np.random.Generator, n: int) -> pd.DataFrame:
    height = np.clip(rng.gamma(4, 11, n), 9, 200)
    age = rng.uniform(0, 35, n)
    trips = np.clip(rng.normal(300 + 5.5 * height, 120), 30, None)
    door_cycles = trips * rng.normal(2.1, 0.2, n)
    wait = np.clip(13 + 0.2 * height + 0.01 * trips + rng.normal(0, 6, n), 5, None)
    mtbf = np.exp(rng.normal(4.2 - 0.03 * age - 0.0005 * trips, 0.5))
    downtime = np.clip(30 / mtbf * rng.gamma(2, 1.5, n), 0, None)
    vibration = np.clip(rng.normal(12 + 0.25 * age, 3), 3, 40)
    return pd.DataFrame({
        "travel_height_m": height.round(1), "installation_age_years": age.round(1), "trips_per_day": trips.round(0),
        "door_cycles_per_day": door_cycles.round(0), "mean_wait_s": wait.round(1),
        "mean_days_between_faults": mtbf.round(1), "downtime_hours": downtime.round(1),
        "car_vibration_mg": vibration.round(1),
    })  # fmt: skip


def _museum(rng: np.random.Generator, n: int) -> pd.DataFrame:
    temp = rng.normal(17, 8, n)
    rain = np.exp(rng.normal(0.5, 1.2, n))
    occupancy = np.clip(rng.normal(58, 17, n), 10, 100)
    visitors = np.clip(
        400 + 35 * occupancy + 60 * np.clip(rain, 0, 15) - 15 * np.abs(temp - 18) + rng.normal(0, 500, n), 150, None
    )  # fmt: skip
    guided = np.clip(visitors * rng.beta(2, 18, n), 0, None)
    dwell = np.clip(rng.normal(95 - 0.004 * (visitors - 2500), 15), 30, 200)
    shop = visitors * np.clip(rng.normal(4.5, 1.2, n), 0.5, None) / 1000
    co2 = np.clip(500 + 0.15 * visitors + rng.normal(0, 60, n), 400, 2500)
    return pd.DataFrame({
        "outside_temp_c": temp.round(1), "rain_mm": rain.round(1), "city_hotel_occupancy_pct": occupancy.round(1),
        "visitors": visitors.round(0), "guided_tour_visitors": guided.round(0), "mean_dwell_min": dwell.round(0),
        "shop_sales_k": shop.round(2), "gallery_co2_ppm": co2.round(0),
    })  # fmt: skip


def _dyehouse(rng: np.random.Generator, n: int) -> pd.DataFrame:
    batch = np.clip(rng.normal(300, 80, n), 80, 600)
    liquor = np.clip(rng.normal(8, 1.5, n), 4, 14)
    temp = np.clip(rng.normal(95, 8, n), 60, 135)
    time = np.clip(rng.normal(70, 15, n) - 0.3 * (temp - 95), 25, 160)
    depth = np.clip(rng.normal(2.0, 0.9, n), 0.1, 6)
    exhaustion = np.clip(70 + 0.15 * (temp - 95) + 0.1 * (time - 70) - 1.2 * (liquor - 8) + rng.normal(0, 4, n), 30, 99)
    delta_e = np.clip(np.exp(rng.normal(-0.4 + 0.03 * (90 - exhaustion), 0.4)), 0.05, None)
    water = batch * liquor * rng.normal(1.05, 0.05, n) / 1000
    return pd.DataFrame({
        "batch_kg": batch.round(0), "liquor_ratio": liquor.round(1), "dye_temp_c": temp.round(0),
        "dye_minutes": time.round(0), "shade_depth_pct": depth.round(2), "dye_exhaustion_pct": exhaustion.round(1),
        "color_delta_e": delta_e.round(2), "water_m3": water.round(2),
    })  # fmt: skip


def _shelter(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = np.clip(rng.gamma(1.5, 2.2, n), 0.1, 18)
    weight = np.clip(np.exp(rng.normal(2.6, 0.6, n)), 1, 70)
    days_shelter = np.clip(rng.gamma(1.8, 12, n) + 2.5 * age, 1, None)
    words = rng.poisson(120, n).astype(float)
    views = rng.poisson(np.clip(300 - 15 * age + 1.5 * days_shelter + 0.5 * words, 10, None), n).astype(float)
    vet_cost = np.clip(60 + 25 * age + rng.gamma(1.5, 40, n), 20, None)
    behaviour = np.clip(rng.normal(3.6 - 0.05 * age, 0.8), 1, 5)
    fee = np.clip(180 - 12 * age + 20 * (behaviour - 3) + rng.normal(0, 25, n), 0, None)
    return pd.DataFrame({
        "animal_age_years": age.round(1), "weight_kg": weight.round(1), "days_in_shelter": days_shelter.round(0),
        "listing_page_views": views, "listing_text_words": words, "vet_costs": vet_cost.round(0),
        "behaviour_score_1_5": behaviour.round(2), "adoption_fee": fee.round(0),
    })  # fmt: skip


DOMAINS_H7 = (
    Domain(
        990201,
        "Orvalho Dairy Cooperative Milk Recording",
        "Monthly milk recording run in 2032 by the fictional Orvalho dairy cooperative to rank cows for breeding and "
        "feeding decisions. A technician weighs the milk of each cow at one milking with a certified meter and takes "
        "a sample analysed at the cooperative laboratory; body weight comes from the walk-over scale and feed intake "
        "from the herd's feeding robot. Each row is one cow on one test day.",
        {
            "cow_age_months": "age of the cow, in months",
            "days_in_milk": "days since the cow last calved, in days",
            "body_weight_kg": "body weight, in kg",
            "milk_kg_day": "milk yield of the test day, in kg per day",
            "fat_pct": "milk fat content, in %",
            "protein_pct": "milk protein content, in %",
            "somatic_cells_k_ml": "somatic cell count of the milk, an udder-health indicator, in thousand cells per mL",
            "dry_matter_intake_kg": "dry matter eaten in the day, in kg",
        },
        "Yield rises in the first two months after calving and then declines; older cows give more milk; fat content "
        "falls as yield rises and protein follows fat; somatic cells rise with age and late in lactation; heavier "
        "and higher-yielding cows eat more.",
        _dairy,
    ),
    Domain(
        990202,
        "Serra Alta Wind Farm Turbine Records",
        "Ten-minute averages exported in 2031 from the control system of the fictional Serra Alta wind farm for a "
        "performance audit. Wind speed comes from the nacelle anemometer, power and rotor speed from the converter, "
        "and temperatures and vibration from the condition-monitoring sensors. Each row is one turbine in one "
        "ten-minute interval.",
        {
            "wind_speed_m_s": "wind speed at hub height, in m/s",
            "ambient_temp_c": "outside air temperature, in degrees Celsius",
            "air_density_kg_m3": "air density computed from temperature and pressure, in kg/m3",
            "active_power_kw": "electrical power delivered, in kW (rated power 2,300 kW)",
            "rotor_speed_rpm": "rotor speed, in revolutions per minute",
            "blade_pitch_deg": "blade pitch angle, in degrees",
            "gearbox_oil_temp_c": "gearbox oil temperature, in degrees Celsius",
            "nacelle_vibration_mm_s": "vibration velocity measured in the nacelle, in mm/s",
        },
        "Power follows an S-shaped curve of wind speed and levels off at the rated power; denser (colder) air gives "
        "slightly more power; rotor speed rises with wind up to about 12 m/s; the blades pitch only above that speed; "
        "gearbox oil heats with power and outside temperature; vibration rises with rotor speed and power.",
        _turbine,
    ),
    Domain(
        990203,
        "Trans-Planalto Freight Trip Ledger",
        "Trip ledger compiled in 2030 by the fictional Trans-Planalto trucking company to renegotiate contracts. "
        "Distance, driving hours and fuel come from the trucks' telematics units, load from the weighbridge ticket, "
        "delay from the customer's receiving log and revenue from the invoice. Each row is one completed trip.",
        {
            "trip_distance_km": "distance driven on the trip, in km",
            "load_tonnes": "cargo weight, in tonnes",
            "driver_experience_years": "driver's years of professional driving",
            "driving_hours": "time spent driving, in hours",
            "fuel_litres": "diesel used on the trip, in litres",
            "unloading_minutes": "time spent unloading at the customers, in minutes",
            "arrival_delay_min": "arrival time minus the booked slot, in minutes (negative = early)",
            "trip_revenue": "amount invoiced for the trip, in currency units",
        },
        "Driving time and fuel grow with distance, and fuel also with the load; unloading takes longer on long trips "
        "and with heavy loads; delays grow with unloading time and are shorter for experienced drivers; revenue "
        "grows with distance and load.",
        _freight,
    ),
    Domain(
        990204,
        "Casa Laguna Hotel Group Stays",
        "Stay records extracted in 2031 from the property-management system of the fictional Casa Laguna hotel group "
        "to study pricing. Booking data come from the reservation engine, spending on extras from the folio at "
        "check-out and the score from the post-stay survey. Each row is one completed reservation.",
        {
            "booking_lead_days": "days between booking and arrival",
            "home_distance_km": "distance between the guest's home address and the hotel, in km",
            "room_size_m2": "floor area of the room booked, in m2",
            "nightly_rate": "average price per night, in currency units",
            "extras_spend": "spending on restaurant, spa and minibar during the stay, in currency units",
            "stay_hours": "time between check-in and check-out, in hours",
            "review_score": "score given in the post-stay survey, from 1 to 10",
            "guest_age_years": "age of the main guest, in years",
        },
        "Larger rooms cost more and get slightly better scores; bookings made far in advance get slightly lower "
        "rates; guests from farther away stay longer; extras spending grows with the length of stay and with the "
        "rate; expensive stays get slightly lower scores.",
        _hotel,
    ),
    Domain(
        990205,
        "Colmeia Viva Apiary Sensor Network",
        "Daily summaries from the sensor network installed in 2032 by the fictional Colmeia Viva beekeepers' "
        "association. Each hive has a scale, a brood-chamber thermometer and an entrance bee counter; colony "
        "strength and mites are assessed in the monthly inspection closest to the day. Each row is one hive on one day.",
        {
            "outside_temp_c": "mean outside temperature of the day, in degrees Celsius",
            "outside_humidity_pct": "mean outside relative humidity, in %",
            "colony_frames_k_bees": "estimated adult bee population, in thousands of bees",
            "brood_temp_c": "temperature in the brood chamber, in degrees Celsius",
            "forager_trips_hour": "bees leaving the entrance per hour of daylight",
            "hive_weight_kg": "weight of the hive, in kg",
            "daily_weight_gain_kg": "change in hive weight over the day, in kg (nectar intake minus consumption)",
            "mites_per_100_bees": "varroa mites counted per 100 bees in the inspection",
        },
        "Colonies are larger in warm weather; strong colonies keep the brood near 34-35 degrees even in cold or hot "
        "weather; forager traffic grows with colony size and warmth; heavier hives belong to larger colonies; daily "
        "weight gain follows forager traffic; mite counts are higher in larger colonies.",
        _hive,
    ),
    Domain(
        990206,
        "Picos Brancos Ski Resort Daily Operations",
        "Operations log kept in 2031 by the fictional Picos Brancos ski resort to plan staffing. Snow and temperature "
        "come from the summit weather station, visits from the ticket gates, queue times from lift cameras and "
        "revenue from the ticketing system. Each row is one operating day.",
        {
            "snowfall_cm": "fresh snow in the previous 24 hours, in cm",
            "snow_base_cm": "snow depth at the base station, in cm",
            "summit_temp_c": "mean summit temperature, in degrees Celsius",
            "valley_hotel_occupancy_pct": "occupancy of the hotels in the valley that night, in %",
            "skier_visits": "number of skier visits counted at the gates",
            "mean_lift_queue_min": "average waiting time at the lifts, in minutes",
            "lift_hours_operated": "total hours of lift operation in the day, summed over the lifts",
            "ticket_revenue_k": "ticket revenue of the day, in thousand currency units",
        },
        "Busy valley hotels and fresh snow bring more visitors, and very cold or warm days fewer; queues grow with "
        "the number of visitors; very heavy snowfall cuts lift hours; the snow base grows with snowfall; revenue "
        "follows visits.",
        _ski,
    ),
    Domain(
        990207,
        "Ponte Nova Used-Vehicle Auction House",
        "Lot records published in 2030 by the fictional Ponte Nova vehicle auction house for dealers. Vehicle data "
        "come from the inspection report, bids and prices from the auction system and the time on the lot from the "
        "yard records. Each row is one vehicle sold at auction.",
        {
            "vehicle_age_years": "age of the vehicle at the sale, in years",
            "odometer_k_km": "odometer reading, in thousand km",
            "engine_litres": "engine displacement, in litres",
            "condition_grade": "inspector's condition grade, from 1 (poor) to 5 (excellent)",
            "online_watchers": "people following the lot in the online auction",
            "reserve_price": "seller's minimum acceptable price, in currency units",
            "hammer_price": "final sale price, in currency units",
            "days_on_lot": "days between arrival at the yard and the sale",
        },
        "Older vehicles have more kilometres, worse condition and lower prices; bigger engines raise the price; "
        "vehicles in better condition attract more online watchers; the sale price follows the reserve and rises "
        "with the number of watchers; much-watched vehicles sell faster.",
        _auction,
    ),
    Domain(
        990208,
        "Montes Claros Permanent Forest Plots",
        "Plot measurements from the 2031 cycle of the fictional Montes Claros forest inventory, used to estimate "
        "timber stocks. Field crews measure trunk diameters with tapes and heights with laser hypsometers; "
        "elevation and slope come from the plot GPS and the stand age from planting records. Each row is one plot.",
        {
            "elevation_m": "plot elevation, in m",
            "slope_deg": "terrain slope, in degrees",
            "stand_age_years": "age of the stand, in years",
            "mean_dbh_cm": "mean trunk diameter at breast height, in cm",
            "dominant_height_m": "height of the tallest trees, in m",
            "trees_per_ha": "number of trees per hectare",
            "basal_area_m2_ha": "total cross-section area of the trunks, in m2 per hectare",
            "litter_depth_cm": "depth of the leaf litter layer, in cm",
        },
        "Older stands have thicker and taller trees and fewer trees per hectare; trees are a little smaller at high "
        "elevation; height grows with diameter and levels off; basal area combines tree count and diameter; litter "
        "is deeper in older stands and thinner on steep slopes.",
        _forest,
    ),
    Domain(
        990209,
        "Rio Manso Basin Gauging Network",
        "Daily readings from the gauging network run in 2030 by the fictional Rio Manso basin agency for flood "
        "warning and water-quality reports. Stage comes from pressure sensors, discharge from the station rating "
        "curve, rainfall from the nearest rain gauge and the quality variables from multiparameter probes. Each "
        "row is one station on one day.",
        {
            "rain_24h_mm": "rainfall over the catchment in the previous 24 hours, in mm",
            "catchment_km2": "area drained by the station, in km2",
            "stage_m": "water level at the station, in m",
            "discharge_m3_s": "river flow, in m3/s",
            "turbidity_ntu": "water turbidity, in NTU",
            "water_temp_c": "water temperature, in degrees Celsius",
            "dissolved_oxygen_mg_l": "dissolved oxygen, in mg/L",
            "conductivity_us_cm": "electrical conductivity of the water, in microsiemens per cm",
        },
        "Water level rises with rain, more in large catchments; flow grows steeply with level and with catchment "
        "size; rain makes the water more turbid and dilutes it (lower conductivity); warmer and more turbid water "
        "holds less oxygen.",
        _river,
    ),
    Domain(
        990210,
        "Nuvem Sul Data Center Rack Telemetry",
        "Hourly rack telemetry exported in 2032 by the fictional Nuvem Sul colocation provider for a cooling study. "
        "Power comes from the rack power units, temperatures from front and rear sensors, utilization and memory "
        "errors from the servers' management controllers. Each row is one rack in one hour.",
        {
            "servers_in_rack": "number of servers installed in the rack",
            "cpu_utilization_pct": "mean processor utilization of the rack, in %",
            "inlet_temp_c": "air temperature at the rack front, in degrees Celsius",
            "rack_power_kw": "electrical power drawn by the rack, in kW",
            "outlet_temp_c": "air temperature at the rack back, in degrees Celsius",
            "fan_speed_pct": "mean server fan speed, in % of maximum",
            "corrected_memory_errors": "memory errors corrected by the servers in the hour",
            "site_pue": "power usage effectiveness of the hall in that hour",
        },
        "Power grows with the number of servers and their utilization; the outlet air is warmer than the inlet by "
        "an amount that grows with power; fans speed up as the outlet heats and as utilization rises; memory errors "
        "are more frequent in hot, crowded racks; the hall uses more overhead energy when the inlet air is warmer.",
        _datacenter,
    ),
    Domain(
        990211,
        "Aeroporto Lagoa Seca Gate Turnarounds",
        "Turnaround records collected in 2031 by the operations centre of the fictional Lagoa Seca airport to set "
        "ground-handling targets. Aircraft and passenger data come from the airline's departure control system, "
        "times from gate sensors and fuel from the fuelling company. Each row is one aircraft turnaround.",
        {
            "passengers": "passengers on the departing flight",
            "load_factor_pct": "share of seats occupied, in %",
            "checked_bags": "number of checked bags loaded",
            "inbound_delay_min": "arrival delay of the incoming flight, in minutes",
            "ground_crew_hours": "staff-hours of ground handling assigned to the turnaround",
            "turnaround_min": "time from on-block to off-block, in minutes",
            "departure_delay_min": "departure delay, in minutes",
            "fuel_uplift_kg": "fuel loaded at the gate, in kg",
        },
        "Bigger and fuller flights carry more bags and need longer turnarounds; more ground staff shortens the "
        "turnaround; departure delays inherit part of the inbound delay and grow with long turnarounds; bigger "
        "aircraft take more fuel.",
        _airport,
    ),
    Domain(
        990212,
        "Torra Fina Specialty Roastery Batch Log",
        "Batch log kept in 2030 by the fictional Torra Fina specialty coffee roastery for quality control. "
        "Temperatures and times come from the roaster's probes, bean moisture from a moisture meter, colour from a "
        "roast-colour analyser and the cupping score from the weekly tasting panel. Each row is one roast batch.",
        {
            "green_batch_kg": "weight of green coffee charged, in kg",
            "bean_moisture_pct": "moisture of the green beans, in %",
            "charge_temp_c": "drum temperature when the beans are charged, in degrees Celsius",
            "roast_minutes": "total roast time, in minutes",
            "drop_temp_c": "bean temperature when the roast ends, in degrees Celsius",
            "weight_loss_pct": "weight lost during roasting, in %",
            "agtron_color": "roast colour reading (lower = darker)",
            "cupping_score": "score given by the tasting panel, from 0 to 100",
        },
        "Larger and wetter batches take longer to roast, and a hotter charge shortens the roast; long roasts end at "
        "higher temperatures; hotter drops lose more weight and give darker colours; the best scores come from "
        "medium colours.",
        _roastery,
    ),
    Domain(
        990213,
        "Costa Brava Norte Offshore Buoy Array",
        "Hourly observations from the fictional Costa Brava Norte buoy array, published in 2031 for marine "
        "forecasting. Waves come from accelerometers, wind and air data from the buoy mast, and currents from an "
        "acoustic profiler; fetch is computed from the wind direction and the coastline. Each row is one buoy in one hour.",
        {
            "wind_speed_m_s": "wind speed at 4 m above the sea, in m/s",
            "fetch_km": "open-water distance over which the wind blows, in km",
            "significant_wave_m": "significant wave height, in m",
            "peak_period_s": "period of the most energetic waves, in seconds",
            "sea_level_pressure_hpa": "air pressure at sea level, in hPa",
            "sea_surface_temp_c": "sea surface temperature, in degrees Celsius",
            "air_temp_c": "air temperature, in degrees Celsius",
            "surface_current_m_s": "surface current speed, in m/s",
        },
        "Waves grow with wind speed and with fetch; higher waves have longer periods; strong winds come with lower "
        "pressure; air temperature follows the sea temperature, a little cooler; currents strengthen with wind.",
        _buoy,
    ),
    Domain(
        990214,
        "Movimento Fitness Club Membership Panel",
        "Monthly membership panel assembled in 2032 by the fictional Movimento fitness club chain to reduce "
        "cancellations. Visits and session lengths come from the turnstiles, classes and training sessions from the "
        "booking app, fees from billing and satisfaction from a monthly in-app question. Each row is one member in "
        "one month.",
        {
            "member_age_years": "age of the member, in years",
            "tenure_months": "months since the member joined",
            "gym_hours_month": "time spent in the club in the month, in hours",
            "mean_session_min": "average time spent per visit, in minutes",
            "group_class_share_pct": "share of the time in the club spent in group classes, in %",
            "monthly_fee": "fee paid in the month, in currency units",
            "personal_training_spend": "spending on personal training in the month, in currency units",
            "satisfaction_0_10": "answer to 'how satisfied are you with the club?', from 0 to 10",
        },
        "Long-standing members spend more time in the club and pay slightly lower fees; older members come a little "
        "less and stay shorter; members on higher fees spend more on personal training; members who spend more time "
        "in the club are more satisfied.",
        _gym,
    ),
    Domain(
        990215,
        "Seguradora Atlas Motor Claims File",
        "Closed motor claims extracted in 2030 by the fictional Atlas insurance company for a reserving study. "
        "Driver and vehicle data come from the policy, impact speed from the crash recorder or the police report, "
        "costs from the assessors and dates from the claims system. Each row is one closed claim.",
        {
            "driver_age_years": "age of the insured driver, in years",
            "vehicle_value": "market value of the vehicle, in currency units",
            "impact_speed_kmh": "estimated speed at impact, in km/h",
            "vehicle_damage": "assessed repair cost of the vehicle, in currency units",
            "medical_costs": "medical costs of the people injured, in currency units",
            "days_to_report": "days between the accident and the claim report",
            "days_to_settle": "days between the report and the settlement",
            "amount_paid": "total amount paid by the insurer, in currency units",
        },
        "Damage grows with impact speed and with the vehicle's value; medical costs rise steeply with speed; older "
        "drivers report a little later; large claims and late reports take longer to settle; the amount paid "
        "follows the total cost minus the deductible.",
        _claims,
    ),
    Domain(
        990216,
        "Banco Horizonte Mortgage Applications",
        "Mortgage applications received in 2031 by the fictional Banco Horizonte, assembled for a fair-lending "
        "review. Income and job tenure come from verified documents, the credit score from the bureau, the home "
        "price from the purchase contract and the rate from the pricing engine. Each row is one application.",
        {
            "annual_income": "verified annual income of the applicants, in currency units",
            "credit_score": "bureau credit score, from 300 to 850",
            "home_price": "purchase price of the home, in currency units",
            "down_payment_pct": "down payment, in % of the price",
            "loan_amount": "requested loan, in currency units",
            "debt_to_income_pct": "monthly debt payments including the new loan, in % of monthly income",
            "offered_rate_pct": "annual interest rate offered, in %",
            "job_tenure_years": "years in the current job",
        },
        "Higher earners have slightly better scores and buy more expensive homes; applicants with better scores put "
        "more money down; the loan is the price minus the down payment; larger loans relative to income raise the "
        "debt ratio; better scores get lower rates and higher debt ratios higher rates.",
        _mortgage,
    ),
    Domain(
        990217,
        "Palco Aberto Live Music Venues",
        "Show records compiled in 2032 by the fictional Palco Aberto network of live-music venues to plan its "
        "calendar. Ticket data come from the ticketing platform, follower counts from the artists' streaming "
        "profiles, bar and merchandise sales from the point-of-sale system and sound levels from the venue meter. "
        "Each row is one show.",
        {
            "venue_capacity": "maximum audience of the venue",
            "artist_followers": "artist's followers on the main streaming platform",
            "ticket_price": "average ticket price, in currency units",
            "tickets_sold": "number of tickets sold",
            "days_on_sale": "days between the start of sales and the show",
            "bar_sales_k": "bar sales during the show, in thousand currency units",
            "merch_sales_k": "merchandise sales, in thousand currency units",
            "peak_sound_db": "peak sound level measured at the mixing desk, in dB",
        },
        "More popular artists charge more and fill a larger share of the venue; tickets sold are limited by "
        "capacity; bar and merchandise sales grow with the audience; bigger venues are slightly louder.",
        _concert,
    ),
    Domain(
        990218,
        "Hemocentro Vale Verde Donor Sessions",
        "Donation sessions recorded in 2031 by the fictional Vale Verde blood centre to review donor eligibility "
        "screening. Age and donation history come from the donor register, weight, pulse, blood pressure and "
        "haemoglobin from the screening station, and volume and time from the collection scale. Each row is one "
        "whole-blood donation.",
        {
            "donor_age_years": "age of the donor, in years",
            "body_weight_kg": "body weight, in kg",
            "hemoglobin_g_dl": "haemoglobin measured before donation, in g/dL",
            "years_since_first_donation": "years since the donor's first donation",
            "pulse_bpm": "pulse before donation, in beats per minute",
            "systolic_bp_mmhg": "systolic blood pressure before donation, in mmHg",
            "collected_ml": "volume of blood collected, in mL",
            "collection_minutes": "time needed to collect the unit, in minutes",
        },
        "Older donors have been donating for longer; experienced donors have slightly lower pulse; blood pressure "
        "rises with age and weight; collection is quicker for donors with higher haemoglobin and slower for those "
        "with a fast pulse; the collected volume is close to the 450 mL target.",
        _donor,
    ),
    Domain(
        990219,
        "Cervejaria Três Pedras Fermentation Tanks",
        "Fermentation records kept in 2030 by the fictional Três Pedras craft brewery to standardize its recipes. "
        "Gravities come from a digital density meter, temperatures from the tank controllers, yeast counts from the "
        "lab microscope and diacetyl and bitterness from the monthly lab panel. Each row is one fermentation batch.",
        {
            "original_gravity": "specific gravity of the wort before fermentation",
            "pitch_rate_m_cells_ml": "yeast added, in million cells per mL",
            "ferment_temp_c": "fermentation temperature, in degrees Celsius",
            "days_to_terminal": "days until the gravity stopped falling",
            "final_gravity": "specific gravity at the end of fermentation",
            "abv_pct": "alcohol by volume, in %",
            "diacetyl_mg_l": "diacetyl (a buttery off-flavour) at packaging, in mg/L",
            "bitterness_ibu": "bitterness, in international bitterness units",
        },
        "Stronger worts take longer to ferment, and warmer tanks and more yeast finish sooner; final gravity rises "
        "with the original gravity; alcohol is set by the drop from original to final gravity; diacetyl is higher "
        "in warm fermentations and falls with more days in the tank.",
        _brewery,
    ),
    Domain(
        990220,
        "Mineração Serra Dourada Haul Fleet",
        "Haul-cycle records collected in 2031 by the fleet management system of the fictional Serra Dourada "
        "open-pit mine. Payload comes from the trucks' on-board scales, distance and speed from GPS, fuel from the "
        "engine controller and tyre temperature from tyre sensors. Each row is one haul cycle.",
        {
            "payload_tonnes": "material carried, in tonnes",
            "ramp_grade_pct": "average grade of the haul ramp, in %",
            "haul_distance_km": "one-way haul distance, in km",
            "cycle_time_min": "time for a full load-haul-dump-return cycle, in minutes",
            "fuel_litres": "diesel used in the cycle, in litres",
            "tyre_temp_c": "maximum tyre temperature, in degrees Celsius",
            "shovel_queue_min": "time waiting at the shovel, in minutes",
            "mean_speed_kmh": "average travel speed, in km/h",
        },
        "Cycles are longer for longer and steeper hauls and heavier loads; fuel grows with distance, grade and "
        "payload; tyres run hotter with heavier loads and longer hauls; speed is the round trip divided by the "
        "travel time.",
        _haul,
    ),
    Domain(
        990221,
        "Terminal Porto Azul Vessel Calls",
        "Vessel-call records compiled in 2030 by the fictional Porto Azul container terminal for its annual "
        "productivity report. Vessel data come from the shipping line's booking, moves and crane hours from the "
        "terminal operating system, waiting times from the port authority and wind from the quay anemometer. Each "
        "row is one vessel call.",
        {
            "ship_capacity_teu": "vessel capacity, in twenty-foot equivalent units",
            "container_moves": "containers loaded and unloaded",
            "crane_hours": "total working hours of the quay cranes on the vessel",
            "moves_per_crane_hour": "average moves per crane per hour",
            "berth_hours": "time alongside the quay, in hours",
            "anchorage_wait_hours": "time waiting at anchor before berthing, in hours",
            "wind_gust_m_s": "maximum wind gust during the call, in m/s",
            "reefer_moves": "refrigerated containers handled",
        },
        "Bigger ships exchange more containers and get more cranes; crane productivity falls slightly when many "
        "cranes work one ship; crane hours are the moves divided by productivity; time at berth is the crane hours "
        "shared among the cranes plus delays; bigger ships wait a little longer at anchor; refrigerated moves are a "
        "small share of all moves.",
        _port,
    ),
    Domain(
        990222,
        "Lumina Games Player Telemetry",
        "Weekly player telemetry exported in 2032 by the fictional mobile-game studio Lumina Games to study "
        "retention. All variables come from the game's analytics events, except spending, which comes from the app "
        "store's payment reports. Each row is one active player in one week.",
        {
            "player_level": "player's level in the game",
            "days_since_install": "days since the game was installed",
            "mean_session_min": "average session length, in minutes",
            "playtime_hours_week": "time played in the week, in hours",
            "friend_list_size": "number of friends in the player's friend list",
            "spend_last_30d": "in-game purchases in the last 30 days, in currency units",
            "win_rate_pct": "share of matches won, in %",
            "median_ping_ms": "median network latency, in milliseconds",
        },
        "Higher-level players installed the game longer ago, play more hours, have larger friend lists and spend "
        "more; players with more friends also spend more; win rate rises a little with level.",
        _gamers,
    ),
    Domain(
        990223,
        "Linha Verde Commuter Rail Trips",
        "Trip records published in 2031 by the fictional Linha Verde commuter rail operator for a punctuality "
        "review. Boardings come from automatic passenger counters, delays and dwell times from the signalling "
        "system, energy from the train meters and temperature from the depot weather station. Each row is one "
        "train trip.",
        {
            "route_km": "length of the route, in km",
            "mean_stop_spacing_km": "average distance between consecutive station stops, in km",
            "boardings": "passengers boarding along the trip",
            "outside_temp_c": "outside temperature, in degrees Celsius",
            "mean_dwell_s": "average time stopped at each station, in seconds",
            "arrival_delay_min": "delay at the final station, in minutes",
            "peak_load_pct": "maximum occupancy along the trip, in % of seated capacity",
            "traction_kwh": "traction energy used, in kWh",
        },
        "Routes with closely spaced stations have more stops; crowded trains stop longer at stations; delays grow "
        "with crowding, long dwells on routes with many stops, and very hot or cold weather; occupancy follows "
        "boardings; energy grows with distance, passengers and stops.",
        _commuter,
    ),
    Domain(
        990224,
        "Recife das Tartarugas Reef Survey",
        "Underwater transect surveys made in 2030 by divers of the fictional Recife das Tartarugas marine park to "
        "monitor reef health. Cover is scored from photo quadrats, fish are counted along the transect, water "
        "temperature comes from dive computers and turbidity from a Secchi disc. Each row is one transect.",
        {
            "transect_depth_m": "depth of the transect, in m",
            "water_temp_c": "water temperature, in degrees Celsius",
            "secchi_turbidity_index": "turbidity index derived from the Secchi disc reading",
            "live_coral_cover_pct": "seabed covered by live coral, in %",
            "macroalgae_cover_pct": "seabed covered by large algae, in %",
            "fish_species_count": "fish species recorded along the transect",
            "fish_biomass_kg_ha": "estimated fish biomass, in kg per hectare",
            "urchins_per_m2": "sea urchins per m2",
        },
        "Coral cover is highest around 8 m and lower in warmer or more turbid water; algae take the space where "
        "coral is scarce and in turbid water; more coral supports more fish species and biomass; urchins are more "
        "abundant where algae are abundant.",
        _reef,
    ),
    Domain(
        990225,
        "Estação Anilhadora Ponta do Vento",
        "Capture records from the autumn 2031 migration season at the fictional Ponta do Vento bird-ringing "
        "station. Ringers measure each bird with rulers and callipers, weigh it on a digital scale and score its fat "
        "visually; time and temperature are logged at capture and stopover length comes from recaptures. Each row "
        "is one first capture of one bird.",
        {
            "wing_chord_mm": "length of the folded wing, in mm",
            "body_mass_g": "body mass, in g",
            "fat_reserve_g": "estimated visible fat reserves, in g",
            "tarsus_mm": "length of the tarsus bone, in mm",
            "capture_hour": "hour of the capture, in decimal hours",
            "air_temp_c": "air temperature at capture, in degrees Celsius",
            "day_of_year": "day of the year of the capture",
            "stopover_days": "days between first and last capture at the station",
        },
        "Bigger birds (longer wings and tarsus) are heavier; heavier birds carry more fat; air warms through the "
        "morning; birds with large fat reserves leave sooner.",
        _banding,
    ),
    Domain(
        990226,
        "Geleira Branca Mass-Balance Stakes",
        "Stake measurements from the 2031 survey of the fictional Geleira Branca glacier, made for a climate "
        "monitoring programme. Glaciologists read ablation stakes and dig snow pits; temperature is interpolated "
        "from automatic weather stations and velocity from repeated GPS surveys of the stakes. Each row is one stake.",
        {
            "stake_elevation_m": "elevation of the stake, in m",
            "aspect_north_index": "how much the slope faces north, from -1 (south) to 1 (north)",
            "summer_mean_temp_c": "mean summer air temperature at the stake, in degrees Celsius",
            "winter_snow_m_we": "winter snow accumulation, in m of water equivalent",
            "debris_cover_cm": "thickness of rock debris on the ice surface, in cm",
            "summer_melt_m_we": "summer melt, in m of water equivalent",
            "annual_balance_m_we": "net annual mass balance, in m of water equivalent",
            "surface_velocity_m_yr": "ice surface velocity, in m per year",
        },
        "It is colder and snowier higher up; melt grows with summer temperature, is lower on north-facing slopes and "
        "under thick debris; the annual balance is winter snow minus summer melt; ice flows a little slower higher up.",
        _glacier,
    ),
    Domain(
        990227,
        "Elevadores Ascende Maintenance Records",
        "Monthly maintenance records assembled in 2032 by the fictional Ascende elevator-maintenance company to plan "
        "inspections. Trips, door cycles and waiting times come from the elevator controllers, faults and downtime "
        "from service tickets and vibration from the annual ride-quality test. Each row is one elevator in one month.",
        {
            "travel_height_m": "vertical travel of the elevator, in m",
            "installation_age_years": "years since installation",
            "trips_per_day": "average trips per day",
            "door_cycles_per_day": "average door openings per day",
            "mean_wait_s": "average passenger waiting time, in seconds",
            "mean_days_between_faults": "mean time between faults over the last year, in days",
            "downtime_hours": "hours out of service in the month",
            "car_vibration_mg": "ride vibration in the car, in milli-g",
        },
        "Taller installations have more trips and longer waits; door cycles follow trips; older and busier elevators "
        "fault more often and are out of service longer; vibration rises with age.",
        _elevator,
    ),
    Domain(
        990228,
        "Museu do Litoral Visitor Counts",
        "Daily visitor records kept in 2030 by the fictional Museu do Litoral to plan staffing and ventilation. "
        "Visitors are counted by door sensors, tours by the booking system, dwell time by ticket scans at entry "
        "and exit, sales by the shop register and CO2 by the gallery air sensors; weather comes from the city "
        "station. Each row is one opening day.",
        {
            "outside_temp_c": "mean outside temperature, in degrees Celsius",
            "rain_mm": "rainfall during opening hours, in mm",
            "city_hotel_occupancy_pct": "occupancy of the city's hotels that day, in % (a proxy for tourism)",
            "visitors": "visitors counted at the doors",
            "guided_tour_visitors": "visitors in guided tours",
            "mean_dwell_min": "average time spent in the museum, in minutes",
            "shop_sales_k": "museum shop sales, in thousand currency units",
            "gallery_co2_ppm": "peak CO2 concentration in the main gallery, in ppm",
        },
        "Busy tourist days and rainy days bring more visitors and very hot or cold days fewer; tours are a small "
        "share of visitors; crowded days have slightly shorter visits; shop sales and gallery CO2 rise with the "
        "number of visitors.",
        _museum,
    ),
    Domain(
        990229,
        "Tinturaria Arco-Íris Dye-House Batches",
        "Batch records kept in 2031 by the fictional Arco-Íris textile dye house for its quality-control system. "
        "Weights come from the batch scale, temperatures and times from the dyeing-machine controller, exhaustion "
        "from the dye-bath spectrophotometer, colour difference from the lab spectrophotometer and water from the "
        "machine meter. Each row is one dye batch.",
        {
            "batch_kg": "fabric weight in the batch, in kg",
            "liquor_ratio": "litres of dye bath per kg of fabric",
            "dye_temp_c": "dyeing temperature, in degrees Celsius",
            "dye_minutes": "dyeing time at temperature, in minutes",
            "shade_depth_pct": "dye applied, in % of fabric weight",
            "dye_exhaustion_pct": "share of the dye taken up by the fabric, in %",
            "color_delta_e": "colour difference from the approved standard (lower = better match)",
            "water_m3": "water used, in m3",
        },
        "Hotter dyeing runs shorter; uptake improves with temperature and time and worsens with more bath per kg; "
        "batches with poor uptake match the standard worse; water use is batch weight times the liquor ratio.",
        _dyehouse,
    ),
    Domain(
        990230,
        "Abrigo Quatro Patas Adoption Records",
        "Animal records exported in 2032 from the case-management system of the fictional Quatro Patas animal "
        "shelter to study adoption times. Age and weight come from the intake exam, page views and listing "
        "text from the adoption website, costs from veterinary invoices and the behaviour score from staff "
        "assessments. Each row is one adopted animal.",
        {
            "animal_age_years": "estimated age of the animal, in years",
            "weight_kg": "body weight, in kg",
            "days_in_shelter": "days between intake and adoption",
            "listing_page_views": "views of the animal's adoption page",
            "listing_text_words": "length of the adoption listing text, in words",
            "vet_costs": "veterinary costs during the stay, in currency units",
            "behaviour_score_1_5": "staff behaviour assessment, from 1 to 5 (5 = very sociable)",
            "adoption_fee": "adoption fee charged, in currency units",
        },
        "Older animals stay longer in the shelter, get fewer page views, cost more in veterinary care and are "
        "adopted for lower fees; animals that stay longer and have longer listings get more views; sociable animals "
        "carry higher fees.",
        _shelter,
    ),
)

assert len(DOMAINS_H7) == 30 and len({d.source_id for d in DOMAINS_H7}) == 30
assert all(len(d.columns) == 8 for d in DOMAINS_H7)
