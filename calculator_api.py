import math
import pandas as pd
import numpy as np
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from urllib.parse import urlencode

API_KEY = "fKumsKDJAjMChPzyFgdd1QFU2L8Js8Pqn7BdfzUo"
EMAIL = "mahantasubhra243@gmail.com"
BASE_CSV_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-tmy-download.csv"

LIFETIME_YEARS = 25
DEGRADATION_RATE = 0.008
CONSUMPTION_GROWTH = 0.02
INFLATION_RATE = 0.05
EXPORT_TARIFF = 3.0
OM_COST_RATE = 1000
INVERTER_COST = 30000
INV_REPL_INTERVAL = 10
DISCOUNT_RATE = 0.08
EMISSION_FACTOR = 0.82
TREE_CO2_PER_YEAR = 10.0

def download_himawari_tmy(lat, lon):
    wkt = f"POINT({lon} {lat})"
    params = {
        "api_key": API_KEY,
        "email": EMAIL,
        "wkt": wkt,
        "names": "tmy-2020",
        "attributes": "air_temperature,ghi,dni,dhi,wind_speed",
        "utc": "false",
        "leap_day": "false",
        "interval": "60"
    }
    url = BASE_CSV_URL + "?" + urlencode(params)
    meta = pd.read_csv(url, nrows=1)
    elev = float(meta["Elevation"].iloc[0])
    df = pd.read_csv(url, skiprows=2)
    df.index = pd.date_range(start="1/1/2020", periods=len(df), freq="60Min")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "temperature" in df.columns:
        df.rename(columns={"temperature": "temp_air"}, inplace=True)
    return df, elev

def estimate_yield_per_kwp(lat, lon):
    tmy, elev = download_himawari_tmy(lat, lon)

    module_params = {
        "pdc0": 550,
        "gamma_pdc": -0.0025,
        "V_oc_ref": 49.85,
        "I_sc_ref": 13.94,
        "V_mp_ref": 41.80,
        "I_mp_ref": 13.16,
        "alpha_sc": 0.0040,
        "beta_oc": -0.0029,
        "cells_in_series": 144,
        "temp_ref": 25,
        "K": 0.05
    }
    inverter_params = {"paco": 3000, "pdc0": 3000}
    temp_model = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

    location = Location(latitude=lat, longitude=lon, tz="Asia/Calcutta", altitude=elev)
    system = PVSystem(surface_tilt=lat, surface_azimuth=180,
                      module_parameters=module_params,
                      inverter_parameters=inverter_params,
                      modules_per_string=2,
                      temperature_model_parameters=temp_model)

    mc = ModelChain(system, location, dc_model="pvwatts", ac_model="pvwatts", aoi_model="ashrae")
    mc.run_model(tmy)
    ac = mc.results.ac
    annual_kwh = ac.sum() / 1000.0
    dc_capacity_kwp = (module_params["pdc0"] * 2) / 1000.0
    return annual_kwh

def parse_state_subsidy(row, kWp):
    text = str(row.get("State Subsidy", "")).lower()
    cap = float(row.get("Max State Subsidy (Rs)", 0))
    if "per kw" in text:
        rate = float(text.split("per")[0].strip())
    else:
        rate = cap / kWp
    return rate, min(cap, rate * kWp)

def get_central_subsidy(kWp):
    if kWp <= 1: return 30000
    if kWp <= 2: return 60000
    return 78000

def calc_npv(rate, cashflows):
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))

def calc_irr(cashflows):
    coefs = np.array(cashflows, dtype=float)
    roots = np.roots(coefs)
    real_x = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    return real_x[0] - 1 if real_x else None

def run_calculator(state: str, mthly: float, latlong: str):
    # 1) Parse "lat,long" string
    try:
        lat_str, lon_str = latlong.split(",")
        lat, lon = float(lat_str.strip()), float(lon_str.strip())
    except Exception:
        return {"error": "Invalid latlong format. Use 'lat,lon' (e.g. '26.44,91.41')."}
    ypk = estimate_yield_per_kwp(lat, lon)
    annual_need = mthly * 12.0
    kWp_req = math.ceil(annual_need / ypk)
    total_yield = ypk * kWp_req

    df = pd.read_csv("subsidy.csv")
    row = df[df["State"].str.lower() == state.lower()]
    if row.empty:
        return {"error": f"No data found for state: {state}"}
    row = row.iloc[0]

    bracket = f"{min(kWp_req, 5)}kW Cost (Rs)"
    cap_cost = float(row.get(bracket, 0))
    cost_per_kwp = cap_cost / min(kWp_req, int(bracket[0]))
    total_cost = cost_per_kwp * kWp_req

    state_rate, state_sub = parse_state_subsidy(row, kWp_req)
    central_sub = get_central_subsidy(kWp_req)
    total_sub = state_sub + central_sub
    net_cost = total_cost - total_sub

    base_tariff = float(row.get("Avg Elec Tariff (₹/kWh)", 0))

    cfs = [-net_cost]
    cum = -net_cost
    payback = None
    yields = []
    for t in range(1, LIFETIME_YEARS + 1):
        yt = total_yield * (1 - DEGRADATION_RATE) ** (t - 1)
        ct = annual_need * (1 + CONSUMPTION_GROWTH) ** (t - 1)
        tt = base_tariff * (1 + INFLATION_RATE) ** (t - 1)
        save_self = ct * tt
        export = max(yt - ct, 0) * EXPORT_TARIFF
        omc = OM_COST_RATE * kWp_req * (1 + INFLATION_RATE) ** (t - 1)
        irc = INVERTER_COST if (t % INV_REPL_INTERVAL == 0) else 0
        cf = (save_self + export) - (omc + irc)
        cfs.append(cf)
        cum += cf
        if payback is None and cum >= 0:
            payback = t
        yields.append(yt)

    npv = calc_npv(DISCOUNT_RATE, cfs)
    irr = calc_irr(cfs)
    life_sav = sum(cfs[1:])
    tot_co2 = sum(yields) * EMISSION_FACTOR
    trees = tot_co2 / TREE_CO2_PER_YEAR

    return {
        "recommended_kw": kWp_req,
        "yield_per_kwp": round(ypk, 1),
        "total_yield_kwh_year1": round(total_yield),
        "total_cost": round(total_cost),
        "state_subsidy": round(state_sub),
        "central_subsidy": round(central_sub),
        "net_cost": round(net_cost),
        "npv": round(npv),
        "irr_percent": round(irr * 100, 2) if irr else None,
        "payback_period_years": payback or ">25",
        "lifetime_savings": round(life_sav),
        "co2_avoided_kg": round(tot_co2),
        "trees_saved_equivalent": round(trees)
    }