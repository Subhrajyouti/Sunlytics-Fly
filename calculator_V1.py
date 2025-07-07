#!/usr/bin/env python3
"""
Residential Solar Calculator
— Uses your original yield script parameters verbatim, with custom IRR
"""
import sys
import math
import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode

from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# ────────────────────────────────────────────────────────────────
# 1) CONFIG & CONSTANTS
# ────────────────────────────────────────────────────────────────

API_KEY      = "fKumsKDJAjMChPzyFgdd1QFU2L8Js8Pqn7BdfzUo"
EMAIL        = "mahantasubhra243@gmail.com"
BASE_CSV_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-tmy-download.csv"

LIFETIME_YEARS     = 25
DEGRADATION_RATE   = 0.008     # 0.8%/yr
CONSUMPTION_GROWTH = 0.02      # 2%/yr
INFLATION_RATE     = 0.05      # 5%/yr
EXPORT_TARIFF      = 3.0       # ₹/kWh exported
OM_COST_RATE       = 1000      # ₹ per kWp·yr
INVERTER_COST      = 30000     # ₹ every 10 yrs
INV_REPL_INTERVAL  = 10
DISCOUNT_RATE      = 0.08      # for NPV/IRR
EMISSION_FACTOR    = 0.82      # kg CO₂/kWh
TREE_CO2_PER_YEAR  = 10.0      # kg CO₂ per tree·yr

# ────────────────────────────────────────────────────────────────
# 2) YIELD ESTIMATION (exactly as your original script)
# ────────────────────────────────────────────────────────────────

def download_himawari_tmy(lat, lon,
                         tmy_version="tmy-2020",
                         interval=60,
                         attributes="air_temperature,ghi,dni,dhi,wind_speed"):
    wkt = f"POINT({lon} {lat})"
    params = {
        "api_key":    API_KEY,
        "email":      EMAIL,
        "wkt":        wkt,
        "names":      tmy_version,
        "attributes": attributes,
        "utc":        "false",
        "leap_day":   "false",
        "interval":   str(interval),
    }
    url = BASE_CSV_URL + "?" + urlencode(params)
    meta = pd.read_csv(url, nrows=1)
    elev = float(meta["Elevation"].iloc[0])
    df   = pd.read_csv(url, skiprows=2)
    year = tmy_version.split("-",1)[-1]
    idx  = pd.date_range(start=f"1/1/{year} 00:00",
                         periods=len(df),
                         freq=f"{interval}Min")
    df.index = idx
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "temperature" in df.columns:
        df.rename(columns={"temperature":"temp_air"}, inplace=True)
    return df, elev

def estimate_yield_per_kwp(lat, lon):
    """
    1) Replicates your standalone script:
       - module_params pdc0=550, modules_per_string=2
       - inverter_params paco=3000,pdc0=3000
       - tilt=lat°, azimuth=180°, same temp model
    2) Runs ModelChain for that ~1.1 kWp system
    3) Divides annual kWh by installed DC kWp to get kWh/kWp·yr
    """
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
    inverter_params = {"paco":3000, "pdc0":3000}
    temp_model = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

    location = Location(latitude=lat,
                        longitude=lon,
                        tz="Asia/Calcutta",
                        altitude=elev)
    system = PVSystem(surface_tilt=lat,
                      surface_azimuth=180,
                      module_parameters=module_params,
                      inverter_parameters=inverter_params,
                      modules_per_string=2,
                      temperature_model_parameters=temp_model)

    mc = ModelChain(system,
                    location,
                    dc_model="pvwatts",
                    ac_model="pvwatts",
                    aoi_model="ashrae")
    mc.run_model(tmy)

    ac = mc.results.ac  # in watts
    annual_kwh = ac.sum() / 1000.0

    # Installed DC capacity in kWp:
    dc_capacity_kwp = (module_params["pdc0"] * 2) / 1000.0

    return annual_kwh 

# ────────────────────────────────────────────────────────────────
# 3) SUBSIDY & COST HELPERS
# ────────────────────────────────────────────────────────────────

def parse_state_subsidy(row, kWp):
    text = str(row.get("State Subsidy", "")).lower()
    cap  = float(row.get("Max State Subsidy (Rs)",0))
    if "per kw" in text:
        rate = float(text.split("per")[0].strip())
    else:
        rate = cap / kWp
    return rate, min(cap, rate * kWp)

def get_central_subsidy(kWp):
    if kWp <= 1: return 30000
    if kWp <= 2: return 60000
    return 78000

# ────────────────────────────────────────────────────────────────
# 4) CUSTOM IRR & NPV
# ────────────────────────────────────────────────────────────────

def calc_npv(rate, cashflows):
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))

def calc_irr(cashflows):
    # cashflows: list of c0, c1, ..., cn
    # find root x of polynomial: sum_{i=0}^n cf[i] * x^(n-i) = 0, where x = 1 + r
    coefs = np.array(cashflows, dtype=float)
    roots = np.roots(coefs)
    # filter real, positive x
    real_x = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not real_x:
        return None
    # choose the first valid root
    return real_x[0] - 1

# ────────────────────────────────────────────────────────────────
# 5) MAIN
# ────────────────────────────────────────────────────────────────

def main():
    state = input("State: ").strip()
    mthly = float(input("Avg monthly consumption (kWh): ").strip())
    raw   = input("Latitude, Longitude (e.g. 26.44,91.41): ").strip()
    try:
        lat, lon = [float(x) for x in raw.split(",")]
    except:
        print("Invalid lat/lon"); sys.exit(1)

    # Yield & sizing
    print("Calculating yield (this may take ~1 min)…")
    ypk = estimate_yield_per_kwp(lat, lon)
    annual_need = mthly * 12.0
    kWp_req     = math.ceil(annual_need / ypk)
    total_yield = ypk * kWp_req

    # Load subsidy/costs
    df = pd.read_csv("subsidy.csv")
    row = df[df["State"].str.lower()==state.lower()]
    if row.empty:
        print(f"No data for {state}"); sys.exit(1)
    row = row.iloc[0]

    # Capital cost lookup
    bracket = f"{min(kWp_req,5)}kW Cost (Rs)"
    cap_cost = float(row.get(bracket,0))
    cost_per_kwp = cap_cost / min(kWp_req,int(bracket[0]))
    total_cost   = cost_per_kwp * kWp_req

    # Subsidies
    state_rate, state_sub = parse_state_subsidy(row, kWp_req)
    central_sub = get_central_subsidy(kWp_req)
    total_sub   = state_sub + central_sub
    net_cost    = total_cost - total_sub

    # Base tariff
    base_tariff = float(row.get("Avg Elec Tariff (₹/kWh)",0))

    # 25-yr cashflows
    cfs = [-net_cost]
    cum = -net_cost
    payback = None
    yields = []
    for t in range(1, LIFETIME_YEARS+1):
        yt = total_yield * (1-DEGRADATION_RATE)**(t-1)
        ct = annual_need  * (1+CONSUMPTION_GROWTH)**(t-1)
        tt = base_tariff * (1+INFLATION_RATE)**(t-1)
        save_self = ct*tt
        export   = max(yt-ct,0)*EXPORT_TARIFF
        omc      = OM_COST_RATE*kWp_req*(1+INFLATION_RATE)**(t-1)
        irc      = INVERTER_COST if (t%INV_REPL_INTERVAL==0) else 0
        cf       = (save_self+export) - (omc+irc)
        cfs.append(cf)
        cum += cf
        if payback is None and cum>=0:
            payback = t
        yields.append(yt)

    # NPV & IRR using our custom functions
    npv = calc_npv(DISCOUNT_RATE, cfs)
    irr = calc_irr(cfs)
    life_sav = sum(cfs[1:])

    # Environmental impact
    tot_co2 = sum(yields) * EMISSION_FACTOR
    trees   = tot_co2 / TREE_CO2_PER_YEAR

    # Output
    print(f"\n— Sizing & Yield —")
    print(f" kWp needed:   {kWp_req} kWp")
    print(f" Yield (yr1):  {total_yield:.0f} kWh (≈{ypk:.1f} kWh/kWp)")

    print(f"\n— Costs & Subsidy —")
    print(f" Install cost: ₹{total_cost:,.0f}")
    print(f" State sub:    ₹{state_sub:,.0f} (₹{state_rate:.0f}/kWp)")
    print(f" Central sub:  ₹{central_sub:,.0f}")
    print(f" Net cost:     ₹{net_cost:,.0f}")

    print(f"\n— Financials (25 yr) —")
    print(f" NPV @{DISCOUNT_RATE*100:.1f}%: ₹{npv:,.0f}")
    if irr is not None:
        print(f" IRR: {irr*100:.2f}%")
    else:
        print(" IRR: could not be determined")
    print(f" Payback: {payback or '>25'} yr")
    print(f" Lifetime savings: ₹{life_sav:,.0f}")

    print(f"\n— Environmental —")
    print(f" CO₂ avoided: {tot_co2:,.0f} kg")
    print(f" Trees saved: {trees:,.0f}")

if __name__=="__main__":
    main()
