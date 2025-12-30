# Copyright 2025 Lukas Neusius
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Simulation der Profitabilität eines Heimspeichers unter Berücksichtigung von:
- Dynamischen Stromtarifen (Day-Ahead)
- Variablen Netzentgelten nach § 14a EnWG (Modul 3)
- Netzentgeltbefreiung für Speicher nach § 118 Abs. 6 EnWG (MiSpeL)
- Degradation (kalendarisch) per "Badewannenkurve"

Das Modell nutzt einen MIP-Solver (Mixed Integer Programming) via CVXPY.
"""
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import cvxpy as cp
import sys

# --- KONFIGURATION & SIMULATIONSPARAMETER ---

# Dateipfade
PRICE_FILE_PATH = 'Day Ahead Dez24-Dez25.csv'
PV_DATA_FILE_PATH = 'zusammengefasste_pv_daten_komplett.csv' 

# Ausgabedateien
LOG_EXPORT_FILE = 'simulation_detailed_log_modul3_3Buck_e8deg02_EXAA.csv'
SUMMARY_EXPORT_FILE = 'simulation_summary_modul3_3Buck_e8deg02_EXAA.txt'

# Technische Daten
BATTERY_CAPACITY_KWH = 71.7
BATTERY_POWER_KW = 20.0           

# Effizienzen (AC-GEKOPPELTES SYSTEM)
AC_EFFICIENCY = 0.955
EFF_IN = AC_EFFICIENCY
EFF_OUT = AC_EFFICIENCY

# Zeit-Parameter
TIME_STEP_HOURS = 0.25           # 15 Minuten

# --- KOSTEN STRUKTUR (§ 14a EnWG Modul 3) ---
# Definition der Preisbestandteile für die Berechnung der Rückerstattungen und Kosten

# Feste Bestandteile (ct/kWh)
LEVIES = 3.50                    # Umlagen (KWKG, Offshore, etc.) - privilegierbar
ELECTRICITY_TAX = 2.05           # Stromsteuer - privilegierbar
CONCESSION_FEE_LOW = 0.61        # Konzessionsabgabe Niedriglast (ct/kWh)
CONCESSION_FEE_STD = 1.32        # Konzessionsabgabe Sonst (ct/kWh)
VAT_RATE = 0.19                  # Mehrwertsteuer (19%)

FEE_NON_REFUND = 2             # Beschaffungskosten / Vertrieb (immer fällig)

# Variable Netzentgelte (§ 14a EnWG Modul 3) netto (Zeitabhängig)
VAR_FEE_LOW = 0.95               # Niedriglasttarifstufe: 00:00 - 07:00
VAR_FEE_STD = 9.53               # Standardlasttarifstufe: Übrige Zeit
VAR_FEE_HIGH = 15.56             # Hochlasttarifstufe: 15:00 - 20:00

# Fixer Tarif Preis und Einspeisevergütung für Basis ohne Speicher
FIXED_TARIFF_LOAD = 0.24         # EUR/kWh Bezugspreis (inkl. Steuern/Abgaben)
FIXED_FEED_IN_TARIFF = 0.076      # EUR/kWh Einspeisevergütung

# --- DEGRADATIONS-MODELL ---
SOC_TARGET = 0.47
COST_SOC_e8 = 0.2

# Simulation
ANNUAL_CONSUMPTION_TARGET_KWH = 9756.0
OPTIMIZATION_HORIZON_HOURS = 34
EXECUTION_HOURS = 24             
OPTIMIZATION_TRIGGER_HOUR = 14   
SIM_START_DATE = '2024-12-26'            # Optional: 'YYYY-MM-DD', z.B. '2025-01-01'
SIM_END_DATE = '2025-12-27'              # Optional: 'YYYY-MM-DD'

# --- 1. HILFSFUNKTIONEN ---

def get_variable_fees(index):
    """
    Erstellt Vektoren für zeitabhängige Netzentgelte und Konzessionsabgaben.
    0-7 Uhr: Low
    15-20 Uhr: High
    Sonst: Std
    """
    grid_fees = []
    concession_fees = []
    for ts in index:
        h = ts.hour
        # Definition der Zeitfenster (Stunden inklusiv/exklusiv prüfen)
        # 0-7 Uhr bedeutet meist bis 07:00 (also Stunde 0, 1, 2, 3, 4, 5, 6)
        if 0 <= h < 7:
            grid_fees.append(VAR_FEE_LOW)
            concession_fees.append(CONCESSION_FEE_LOW)
        # 15-20 Uhr bedeutet meist bis 20:00 (also 15, 16, 17, 18, 19)
        elif 15 <= h < 20:
            grid_fees.append(VAR_FEE_HIGH)
            concession_fees.append(CONCESSION_FEE_STD)
        else:
            grid_fees.append(VAR_FEE_STD)
            concession_fees.append(CONCESSION_FEE_STD)
    return np.array(grid_fees), np.array(concession_fees)

def generate_synthetic_load(index, pv_power_curve, total_kwh):
    if len(index) == 0: return pd.Series([], index=index)
    hour_of_day = index.hour.values + index.minute.values / 60.0
    
    # 1. Haushalt
    base_profile = 0.2 + 0.15 * np.exp(-((hour_of_day - 10)**2)/8) + 0.2 * np.exp(-((hour_of_day - 19)**2)/6)
    noise = np.random.normal(0, 0.05, len(index))
    household_load = np.maximum(base_profile + noise, 0.1)
    
    # 2. Wärmepumpe
    wp_target_annual_kwh = 6219.0
    day_of_year = index.dayofyear.values
    seasonal_factor = 0.5 * (1 + np.cos((day_of_year - 20) * 2 * np.pi / 365)) 
    daily_wp_factor = 1.0 + 0.4 * np.cos((hour_of_day - 5) * 2 * np.pi / 24) 
    wp_load_raw = seasonal_factor * daily_wp_factor * np.random.uniform(0.5, 1.5, len(index))
    
    # Skalierung auf Zielverbrauch WP
    sim_years = (len(index) * TIME_STEP_HOURS) / (365 * 24)
    if sim_years > 0:
        target_sum_wp = wp_target_annual_kwh * sim_years
        current_sum_wp = wp_load_raw.sum() * TIME_STEP_HOURS
        if current_sum_wp > 0:
            wp_load = wp_load_raw * (target_sum_wp / current_sum_wp)
        else:
            wp_load = wp_load_raw
    else:
        wp_load = wp_load_raw
    
    # 3. E-Auto
    ev_load = np.zeros(len(index))
    unique_days = np.unique(index.date)
    
    ev_target_annual_kwh = 1200.0
    avg_charge_kwh = 15
    daily_charge_prob = (ev_target_annual_kwh / avg_charge_kwh) / 365.0
    daily_charge_prob = np.clip(daily_charge_prob, 0.0, 1.0)
    
    for d in unique_days:
        day_mask = (index.date == d)
        day_pv = pv_power_curve[day_mask]
        
        if np.random.rand() < daily_charge_prob: 
            charge_need_kwh = np.random.uniform(15, 30)
            if day_pv.sum() > 0: 
                 total_day_pv_energy = day_pv.sum() * TIME_STEP_HOURS
            else:
                 total_day_pv_energy = 0
            
            if total_day_pv_energy > 10.0: 
                charge_profile = day_pv / day_pv.sum()
                ev_power_curve = np.clip(charge_profile * charge_need_kwh / TIME_STEP_HOURS, 0, 11.0)
            else:
                start_step = np.random.randint(40, 60) 
                if start_step < len(day_pv):
                    duration_steps = int(charge_need_kwh / 11.0 / TIME_STEP_HOURS)
                    end_step = min(start_step + duration_steps, len(day_pv))
                    ev_power_curve = np.zeros(len(day_pv))
                    ev_power_curve[start_step:end_step] = 11.0 
                else:
                    ev_power_curve = np.zeros(len(day_pv))
            ev_load[day_mask] += ev_power_curve

    total_raw_kw = household_load + wp_load + ev_load
    if total_raw_kw.sum() == 0: scaling_factor = 1
    else: scaling_factor = total_kwh / (total_raw_kw.sum() * TIME_STEP_HOURS)
    
    return pd.Series(total_raw_kw * scaling_factor, index=index, name='Load_kW')

def load_data():
    print("Lade Daten...")
    try:
        # 1. PREIS DATEN
        df_price = pd.read_csv(PRICE_FILE_PATH, skiprows=2, header=None, names=['TimestampStr', 'Price_MWh'], decimal='.') 
        
        # ISO-Format direkt parsen (UTC) und in lokale Zeitzone (Berlin) konvertieren
        df_price['Timestamp'] = pd.to_datetime(df_price['TimestampStr'], utc=True)
        df_price['Timestamp'] = df_price['Timestamp'].dt.tz_convert('Europe/Berlin')
        
        df_price = df_price.dropna(subset=['Timestamp']).set_index('Timestamp').sort_index()
        
        df_price['DayAhead_EUR_kWh'] = df_price['Price_MWh'] / 1000.0
        
        # --- PREIS VEKTOREN BAUEN ---
        grid_fees, conc_fees = get_variable_fees(df_price.index)
        df_price['Var_Grid_Fee_Cent'] = grid_fees
        df_price['Concession_Fee_Cent'] = conc_fees
        
        # --- PREISVEKTOREN FÜR OPTIMIERUNG ---
        # Voller Bezugspreis (für Optimierung "Greedy" Ansatz, Rückerstattung erfolgt ex-post)
        # Preis = Spot + Netzentgelt + Konzession + Umlagen + Stromsteuer + Beschaffung
        full_levies = (df_price['Var_Grid_Fee_Cent'] + df_price['Concession_Fee_Cent'] + LEVIES + ELECTRICITY_TAX + FEE_NON_REFUND) / 100.0
        
        df_price['Price_Buy_Full'] = (df_price['DayAhead_EUR_kWh'] + full_levies) * (1 + VAT_RATE)
        
        # Arbitrage-Preis: Spot + NonRefund + Kosten für Verluste (Gebühren + MwSt)
        # Dieser Preis wird vom Solver genutzt, um zu entscheiden, ob sich Arbitrage lohnt.
        # Verluste (1 - eta^2) gelten als Letztverbrauch -> Volle Abgaben (außer Umlagen) + MwSt
        loss_factor = 1.0 - (AC_EFFICIENCY * AC_EFFICIENCY)
        
        # Gebühren auf Verluste (Grid, Conc, Tax). Umlagen (LEVIES) sind privilegiert.
        fees_losses = (df_price['Var_Grid_Fee_Cent'] + df_price['Concession_Fee_Cent'] + ELECTRICITY_TAX) / 100.0
        
        # Wert der Verluste für MwSt (Spot + NonRefund + Fees)
        val_losses = df_price['DayAhead_EUR_kWh'] + (FEE_NON_REFUND / 100.0) + fees_losses
        
        # Preisvektor: Basis (Spot+NonRefund) + Anteilige Gebühren für Verluste + MwSt auf Verluste
        df_price['Price_Buy_Arb'] = (df_price['DayAhead_EUR_kWh'] + FEE_NON_REFUND / 100.0) + \
                                    (loss_factor * fees_losses) + \
                                    (loss_factor * val_losses * VAT_RATE)
        
        df_price = df_price.drop(columns=['TimestampStr', 'Price_MWh'], errors='ignore')
        
        # 2. PV DATEN
        # Datei hat Header (Zeitraum, Leistung [kW]), daher header=0
        df_pv = pd.read_csv(PV_DATA_FILE_PATH, header=0, decimal='.')
        
        # Spalten umbenennen für einheitliche Weiterverarbeitung
        df_pv = df_pv.rename(columns={'Zeitraum': 'Timestamp', 'Leistung [kW]': 'PV_kW'})
        
        # ISO-Format parsen und Zeitzone anpassen
        df_pv['Timestamp'] = pd.to_datetime(df_pv['Timestamp'], utc=True)
        df_pv['Timestamp'] = df_pv['Timestamp'].dt.tz_convert('Europe/Berlin')
        
        df_pv = df_pv.dropna(subset=['Timestamp']).set_index('Timestamp').sort_index()
        df_pv['PV_kW'] = df_pv['PV_kW'].clip(lower=0.0)

    except Exception as e: 
        print(f"Fehler beim Laden der CSVs: {e}")
        return None

    # Join der Daten (beide Indizes sind jetzt 'Europe/Berlin')
    df = df_pv.join(df_price, how='outer', lsuffix='_pv', rsuffix='_price')
    
    # --- ZEITRAUM EINSCHRÄNKEN ---
    # 1. Nur Zeitraum mit Day Ahead Daten (Schnittmenge)
    if not df_price.empty:
        df = df.loc[df_price.index.min() : df_price.index.max()]

    # 2. Manuelle Grenzen
    if SIM_START_DATE:
        df = df.loc[SIM_START_DATE:]
    if SIM_END_DATE:
        df = df.loc[:SIM_END_DATE]

    df['DayAhead_EUR_kWh'] = df['DayAhead_EUR_kWh'].ffill().bfill()
    df['Price_Buy_Full'] = df['Price_Buy_Full'].ffill().bfill()
    df['Price_Buy_Arb'] = df['Price_Buy_Arb'].ffill().bfill()
    df['Var_Grid_Fee_Cent'] = df['Var_Grid_Fee_Cent'].ffill().bfill()
    df['Concession_Fee_Cent'] = df['Concession_Fee_Cent'].ffill().bfill()
    df['PV_kW'] = df['PV_kW'].fillna(0)
    
    total_sim_hours = (df.index.max() - df.index.min()).total_seconds() / 3600
    if total_sim_hours > 0:
        target_load = ANNUAL_CONSUMPTION_TARGET_KWH * (total_sim_hours / (365*24))
    else: target_load = 100
        
    if 'Load_kW' not in df.columns:
        df['Load_kW'] = generate_synthetic_load(df.index, df['PV_kW'].values, target_load)
    
    print(f"Daten bereit. Zeitraum: {df.index.min()} bis {df.index.max()}")
    return df

# --- 2. OPTIMIERUNG MIT CVXPY (Modul 3 Logic) ---

def optimize_window_cvxpy(df_window, initial_soc_grey_load, initial_soc_grey_arb, initial_soc_green):
    N = len(df_window)
    
    p_buy_full = df_window['Price_Buy_Full'].values 
    p_buy_arb = df_window['Price_Buy_Arb'].values
    p_sell = df_window['DayAhead_EUR_kWh'].values   
    
    pv = df_window['PV_kW'].values
    load = df_window['Load_kW'].values
    
    # --- VARIABLEN ---
    # Grid -> Battery (Split in Load-Bucket und Arbitrage-Bucket)
    # Wir unterscheiden, wofür der Strom aus dem Netz geladen wird:
    # 1. Load-Bucket: Strom, der später im Haus verbraucht wird (spart vollen Bezugspreis).
    # 2. Arbitrage-Bucket: Strom, der zurück ins Netz gespeist wird (nutzt Preisspreads).
    grid2batt_load = cp.Variable(N, nonneg=True) 
    grid2batt_arb = cp.Variable(N, nonneg=True)
    
    pv2batt = cp.Variable(N, nonneg=True)   
    
    batt2load_grey_load = cp.Variable(N, nonneg=True) 
    
    batt2grid_grey_arb = cp.Variable(N, nonneg=True) # Arbitrage-Bucket darf nur ins Netz entladen
    
    batt2load_green = cp.Variable(N, nonneg=True) 
    batt2grid_green = cp.Variable(N, nonneg=True) 
    
    grid2load = cp.Variable(N, nonneg=True) 
    pv2load = cp.Variable(N, nonneg=True)   
    pv2grid = cp.Variable(N, nonneg=True)   
    
    soc_grey_load = cp.Variable(N+1, nonneg=True)
    soc_grey_arb = cp.Variable(N+1, nonneg=True)
    soc_green = cp.Variable(N+1, nonneg=True)
    
    # Binäre Variable für den Status: 1 = Laden, 0 = Entladen/Ruhe
    is_charging = cp.Variable(N, boolean=True)
    
    constraints = []
    cost_terms = []
    
    constraints.append(soc_grey_load[0] == initial_soc_grey_load)
    constraints.append(soc_grey_arb[0] == initial_soc_grey_arb)
    constraints.append(soc_green[0] == initial_soc_green)
    
    for t in range(N):
        # 1. Bilanz Haushalt
        constraints.append(
            grid2load[t] + pv2load[t] + batt2load_grey_load[t] + batt2load_green[t] == load[t]
        )
        
        # 2. Bilanz PV
        constraints.append(pv2load[t] + pv2batt[t] + pv2grid[t] == pv[t])
        
        # 3. Batterie Dynamik
        # Wir modellieren den Speicher als drei virtuelle "Eimer" (Buckets), um die Stromherkunft zu tracken:
        # - Grey Load: Netzstrom für Eigenverbrauch
        # - Grey Arb: Netzstrom für Rückspeisung
        # - Green: PV-Strom (vorrangig Eigenverbrauch, Überschuss Einspeisung)
        # Bucket: Grey Load
        constraints.append(
            soc_grey_load[t+1] == soc_grey_load[t] + 
            (grid2batt_load[t] * EFF_IN * TIME_STEP_HOURS) - 
            ((batt2load_grey_load[t]) / EFF_OUT * TIME_STEP_HOURS)
        )
        # Bucket: Grey Arbitrage
        constraints.append(
            soc_grey_arb[t+1] == soc_grey_arb[t] + 
            (grid2batt_arb[t] * EFF_IN * TIME_STEP_HOURS) - 
            ((batt2grid_grey_arb[t]) / EFF_OUT * TIME_STEP_HOURS)
        )
        # Bucket: Green
        constraints.append(
            soc_green[t+1] == soc_green[t] + 
            (pv2batt[t] * EFF_IN * TIME_STEP_HOURS) - 
            ((batt2load_green[t] + batt2grid_green[t]) / EFF_OUT * TIME_STEP_HOURS)
        )
        
        # 4. Limits
        total_charge = grid2batt_load[t] + grid2batt_arb[t] + pv2batt[t]
        total_discharge = batt2load_grey_load[t] + batt2grid_grey_arb[t] + batt2load_green[t] + batt2grid_green[t]
        
        constraints.append(total_charge <= BATTERY_POWER_KW * is_charging[t])
        constraints.append(total_discharge <= BATTERY_POWER_KW * (1 - is_charging[t]))
        constraints.append(soc_grey_load[t+1] + soc_grey_arb[t+1] + soc_green[t+1] <= BATTERY_CAPACITY_KWH)
        
        # --- KOSTENFUNKTION ---
        
        # A. Einkauf
        # Haushalt + Speicher(Load) -> Voller Preis
        cost_terms.append((grid2load[t] + grid2batt_load[t]) * p_buy_full[t] * TIME_STEP_HOURS)
        # Speicher(Arbitrage) -> Reduzierter Preis (Spot + NonRefund)
        cost_terms.append((grid2batt_arb[t]) * p_buy_arb[t] * TIME_STEP_HOURS)
        
        # B. Einnahmen
        export_power = pv2grid[t] + batt2grid_grey_arb[t] + batt2grid_green[t]
        cost_terms.append(-export_power * p_sell[t] * TIME_STEP_HOURS)
        
        # C. Quadratische SoC Strafe (Zyklen schonen / Mitte halten)
        soc_total = (soc_grey_load[t+1] + soc_grey_arb[t+1] + soc_green[t+1])/BATTERY_CAPACITY_KWH
        cost_terms.append(COST_SOC_e8 * cp.power(1.55*(soc_total - SOC_TARGET), 8))

    objective = cp.Minimize(cp.sum(cost_terms))
    problem = cp.Problem(objective, constraints)
    
    # MIP Solver Aufruf (benötigt z.B. SCIP, Gurobi, CPLEX, da MIQP durch quadratische Kosten)
    problem.solve(solver=cp.SCIP)

    res = pd.DataFrame(index=df_window.index)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"WARNUNG: Solver konnte keine optimale Lösung für das Fenster ab {df_window.index[0]} finden. Status: {problem.status}. Fallback auf Null-Betrieb.")
        # Create a zero-filled dataframe with all expected columns to avoid crashing the main loop
        zero_values = np.zeros(N)
        res['Opt_Grid2Load_kW'] = zero_values
        res['Opt_PV2Load_kW'] = zero_values
        res['Opt_PV2Grid_kW'] = zero_values
        res['Opt_Grid2Batt_Load_kW'] = zero_values
        res['Opt_Grid2Batt_Arb_kW'] = zero_values
        res['Opt_Grid2Batt_kW'] = zero_values # Summe
        res['Opt_PV2Batt_kW'] = zero_values
        res['Opt_Batt2Load_kW'] = zero_values
        res['Opt_Batt2Grid_kW'] = zero_values
        res['Opt_SoC_End_Grey_Load'] = np.full(N, initial_soc_grey_load)
        res['Opt_SoC_End_Grey_Arb'] = np.full(N, initial_soc_grey_arb)
        res['Opt_SoC_End_Grey'] = res['Opt_SoC_End_Grey_Load'] + res['Opt_SoC_End_Grey_Arb']
        res['Opt_SoC_End_Green'] = np.full(N, initial_soc_green)
        res['Opt_SoC_End_kWh'] = res['Opt_SoC_End_Grey'] + res['Opt_SoC_End_Green'] # Summe
        for col in ['Log_Cost_Import_EUR', 'Log_Revenue_Export_EUR']:
            res[col] = zero_values
        return res

    res['Opt_Grid2Load_kW'] = grid2load.value
    res['Opt_PV2Load_kW'] = pv2load.value
    res['Opt_PV2Grid_kW'] = pv2grid.value
    
    res['Opt_Grid2Batt_Load_kW'] = grid2batt_load.value
    res['Opt_Grid2Batt_Arb_kW'] = grid2batt_arb.value
    res['Opt_Grid2Batt_kW'] = res['Opt_Grid2Batt_Load_kW'] + res['Opt_Grid2Batt_Arb_kW']
    
    res['Opt_PV2Batt_kW'] = pv2batt.value
    res['Opt_Batt2Load_kW'] = batt2load_grey_load.value + batt2load_green.value
    
    res['Opt_Batt2Grid_kW'] = batt2grid_grey_arb.value + batt2grid_green.value
    
    res['Opt_SoC_End_Grey_Load'] = soc_grey_load.value[1:]
    res['Opt_SoC_End_Grey_Arb'] = soc_grey_arb.value[1:]
    res['Opt_SoC_End_Grey'] = res['Opt_SoC_End_Grey_Load'] + res['Opt_SoC_End_Grey_Arb']
    
    res['Opt_SoC_End_Green'] = soc_green.value[1:]
    res['Opt_SoC_End_kWh'] = res['Opt_SoC_End_Grey'] + res['Opt_SoC_End_Green']
    
    # --- KOSTEN-REPORTING ---
    # Vorläufige Kosten (Brutto, ohne Rückerstattung) - Hier loggen wir die "echten" Kosten, die anfallen würden (Full Price)
    # Damit die Ex-Post Rechnung stimmt. Der Solver hat mit reduzierten Kosten optimiert, aber die Rechnung ist Brutto.
    c_import = (grid2load.value + grid2batt_load.value + grid2batt_arb.value) * p_buy_full * TIME_STEP_HOURS
    c_export = (pv2grid.value + batt2grid_grey_arb.value + batt2grid_green.value) * p_sell * TIME_STEP_HOURS
    
    res['Log_Cost_Import_EUR'] = c_import
    res['Log_Revenue_Export_EUR'] = c_export
    
    return res

# --- 3. HAUPTPROGRAMM ---

if __name__ == "__main__":
    df = load_data()
    if df is not None and not df.empty:
        sim_start = df.index.min().normalize() + timedelta(hours=OPTIMIZATION_TRIGGER_HOUR)
        sim_end = df.index.max()
        last_start = sim_end - timedelta(hours=OPTIMIZATION_HORIZON_HOURS)
        days_sim = (last_start - sim_start).days + 1
        
        print(f"Starte MIP-Simulation (Modul 3 Logic) für {days_sim} Tage...")
        
        curr_time = sim_start
        curr_soc_grey_load = 0.0
        curr_soc_grey_arb = 0.0
        curr_soc_green = 0.0
        results = []
        
        try:
            for d in range(days_sim):
                if d % 5 == 0: print(f"Tag {d+1}/{days_sim}...")
                window_end = curr_time + timedelta(hours=OPTIMIZATION_HORIZON_HOURS)
                df_win = df.loc[curr_time : window_end - timedelta(minutes=15)].copy()
                if df_win.empty: break
                
                res_df = optimize_window_cvxpy(df_win, curr_soc_grey_load, curr_soc_grey_arb, curr_soc_green)
                steps = int(EXECUTION_HOURS / TIME_STEP_HOURS)
                exec_slice = res_df.iloc[:steps].copy()
                
                if len(exec_slice) > 0:
                    results.append(exec_slice)
                    curr_soc_grey_load = exec_slice['Opt_SoC_End_Grey_Load'].iloc[-1]
                    curr_soc_grey_arb = exec_slice['Opt_SoC_End_Grey_Arb'].iloc[-1]
                    curr_soc_green = exec_slice['Opt_SoC_End_Green'].iloc[-1]
                    curr_time += timedelta(hours=EXECUTION_HOURS)
        except ImportError:
            sys.exit(1)
        
        if results:
            full = pd.concat(results)
            df_final = df.loc[full.index].join(full)
            
            # --- MiSpeL / EnWG § 118 BILANZIERUNG (Anlage 1, Fall A1) ---
            print("\nBerechne Bilanz nach MiSpeL (Anlage 1, Fall A1)...")
            # Die regulatorische Abrechnung erfolgt ex-post.
            # Es wird ermittelt, welcher Anteil des Speicherstroms aus dem Netz stammt ("Netzstromanteil")
            # und welcher Anteil der Ausspeisung wieder ins Netz ging.
            
            # 1. Definition der Zählwerte gemäß Anlage 1
            # Z1NB: Netzbezug (Grid2Load + Grid2Batt)
            z1nb = df_final['Opt_Grid2Load_kW'] + df_final['Opt_Grid2Batt_kW']
            # Z1NE: Netzeinspeisung (PV2Grid + Batt2Grid)
            z1ne = df_final['Opt_PV2Grid_kW'] + df_final['Opt_Batt2Grid_kW']
            # Z2V: Verbrauch im Speicher (Ladung)
            z2v = df_final['Opt_Grid2Batt_kW'] + df_final['Opt_PV2Batt_kW']
            # Z2E: Erzeugung im Speicher (Entladung)
            z2e = df_final['Opt_Batt2Load_kW'] + df_final['Opt_Batt2Grid_kW']
            
            # --- Analyse der Vorrangregel-Effekte ---
            # Physikalischer Netzbezug für Speicher (vom Solver geplant, z.B. für Arbitrage)
            physical_grid_charge = df_final['Opt_Grid2Batt_kW'].sum() * TIME_STEP_HOURS
            
            # 2. Formeln Anlage 1
            # (1) Zeitgleicher Netzstromverbrauch im Speicher
            simul_grid_cons_storage = np.minimum(z1nb, z2v)
            
            # (2) Zeitgleiche Netzeinspeisung aus Speicher
            simul_grid_feedin_storage = np.minimum(z1ne, z2e)
            
            # Summen über das Jahr (oder Simulationszeitraum)
            sum_simul_grid_cons = simul_grid_cons_storage.sum() * TIME_STEP_HOURS # (6)
            sum_simul_grid_feedin = simul_grid_feedin_storage.sum() * TIME_STEP_HOURS # (7)
            sum_z2v = z2v.sum() * TIME_STEP_HOURS # (4)
            sum_z2e = z2e.sum() * TIME_STEP_HOURS # (5)
            
            # (9) Gesamtverbrauch Speicher (hier ohne Fremdtankstrom)
            total_storage_cons = sum_z2v
            
            # (10) Netzstromanteil
            if total_storage_cons > 0:
                grid_share = sum_simul_grid_cons / total_storage_cons
            else:
                grid_share = 0.0
            
            # (11) Saldierungsfähige Netzeinspeisung (Menge für Rückerstattung)
            refundable_export_kwh = grid_share * sum_simul_grid_feedin
            
            # (12) Verluste im Speicher
            total_losses = sum_z2v - sum_z2e
            
            # (13) Privilegierungsfähige Speicherverluste
            privileged_losses_kwh = grid_share * total_losses
            
            # --- FINANZIELLE RÜCKERSTATTUNG ---
            # Wir berechnen die gezahlten Entgelte auf den Netzbezug für den Speicher
            # und ziehen davon die Rückerstattung ab.
            
            # Gezahlte Umlagen/Steuern auf Grid2Batt (im 'Price_Buy_Full' enthalten)
            # Hinweis: Da wir ex-post bilanzieren, nehmen wir Durchschnittswerte oder summieren exakt auf.
            # Hier summieren wir exakt auf, was für Grid2Batt gezahlt wurde.
            
            # Gezahlte Komponenten für Grid2Batt
            paid_grid_fees = (df_final['Opt_Grid2Batt_kW'] * df_final['Var_Grid_Fee_Cent'] / 100.0).sum() * TIME_STEP_HOURS
            paid_conc_fees = (df_final['Opt_Grid2Batt_kW'] * df_final['Concession_Fee_Cent'] / 100.0).sum() * TIME_STEP_HOURS
            paid_levies = (df_final['Opt_Grid2Batt_kW'] * LEVIES / 100.0).sum() * TIME_STEP_HOURS
            paid_el_tax = (df_final['Opt_Grid2Batt_kW'] * ELECTRICITY_TAX / 100.0).sum() * TIME_STEP_HOURS
            
            # Rückerstattung berechnen (Mengen * Durchschnittssätze oder pauschal, da Zeitgleichheit bei Rückspeisung entkoppelt ist)
            # Vereinfachung: Wir nutzen gewichtete Durchschnittspreise der Einspeicherung für die Rückerstattung,
            # da die Rückerstattung "die gezahlten Entgelte" betrifft.
            
            avg_grid_fee_paid = paid_grid_fees / (df_final['Opt_Grid2Batt_kW'].sum() * TIME_STEP_HOURS) if df_final['Opt_Grid2Batt_kW'].sum() > 0 else 0
            avg_conc_fee_paid = paid_conc_fees / (df_final['Opt_Grid2Batt_kW'].sum() * TIME_STEP_HOURS) if df_final['Opt_Grid2Batt_kW'].sum() > 0 else 0
            
            # Rückerstattung Netzentgelte & Konzessionsabgabe: Nur auf saldierungsfähige Netzeinspeisung
            refund_grid_fees = refundable_export_kwh * avg_grid_fee_paid
            refund_conc_fees = refundable_export_kwh * avg_conc_fee_paid
            
            # Rückerstattung Umlagen & Stromsteuer: Auf saldierungsfähige Einspeisung + privilegierte Verluste
            # (Gemäß § 118 Abs. 6 EnWG neu i.V.m. § 21 EnFG)
            refund_levies = (refundable_export_kwh + privileged_losses_kwh) * (LEVIES / 100.0)
            refund_el_tax = (refundable_export_kwh) * (ELECTRICITY_TAX / 100.0)
            
            total_refund = refund_grid_fees + refund_conc_fees + refund_levies + refund_el_tax
            
            # --- GESAMTKOSTEN ---
            # Kosten aus Simulation (Brutto)
            cost_gross = df_final['Log_Cost_Import_EUR'].sum()
            revenue = df_final['Log_Revenue_Export_EUR'].sum()
            
            # Mehrwertsteuer-Korrektur:
            # Die MwSt (19%) fällt nur für den tatsächlichen Letztverbrauch an.
            # Da in der Abrechnung die Kosten zunächst brutto (inkl. MwSt) berechnet wurden,
            # muss die MwSt auf den erstatteten Teil (der kein Letztverbrauch ist) herausgerechnet werden.
            # Dies betrifft sowohl die erstatteten Netzentgelte als auch den Warenwert (Spotpreis) des durchgeleiteten Stroms.
            
            # Durchschnittlicher gezahlter Spotpreis für Ladung
            paid_spot = (df_final['Opt_Grid2Batt_kW'] * df_final['DayAhead_EUR_kWh']).sum() * TIME_STEP_HOURS
            avg_spot_paid = paid_spot / (df_final['Opt_Grid2Batt_kW'].sum() * TIME_STEP_HOURS) if df_final['Opt_Grid2Batt_kW'].sum() > 0 else 0
            
            refund_vat_spot = refundable_export_kwh * avg_spot_paid * VAT_RATE
            
            refund_vat = (total_refund * VAT_RATE) + refund_vat_spot
            
            cost_net_after_refund = cost_gross - revenue - total_refund - refund_vat
            
            # Referenz ohne Batterie
            net_load = df_final['Load_kW'] - df_final['PV_kW']
            imp_no_batt = np.maximum(0, net_load)
            exp_no_batt = np.maximum(0, -net_load)
            # Hier nehmen wir vereinfacht den fixen Tarif für die Referenz
            cost_base = (imp_no_batt * FIXED_TARIFF_LOAD - exp_no_batt * FIXED_FEED_IN_TARIFF).sum() * TIME_STEP_HOURS

            summary = f"""
=== ERGEBNIS (Modul 3 MIP Simulation) ===
Zeitraum: {df_final.index.min()} bis {df_final.index.max()}
----------------------------------------
Kosten ohne Batterie und festem Tarif: {cost_base:.2f} EUR
----------------------------------------
Kosten mit Batterie (Brutto vor Erstattung): {cost_gross - revenue:.2f} EUR

--- MiSpeL / EnWG § 118 Abrechnung ---
Gesamtladung Speicher (Z2V): {sum_z2v:.2f} kWh

Davon physikalisch aus Netz (Solver): {physical_grid_charge:.2f} kWh
Davon regulatorisch aus Netz (Z1NB n Z2V): {sum_simul_grid_cons:.2f} kWh
-> Differenz durch Vorrangregel: {sum_simul_grid_cons - physical_grid_charge:.2f} kWh

=> Regulatorischer Netzstromanteil: {grid_share*100:.2f} %

Gesamtentladung Speicher (Z2E): {sum_z2e:.2f} kWh
Davon ins Netz (Z1NE n Z2E): {sum_simul_grid_feedin:.2f} kWh

=> Saldierungsfähige Netzeinspeisung: {refundable_export_kwh:.2f} kWh
=> Privilegierte Speicherverluste: {privileged_losses_kwh:.2f} kWh

Rückerstattungen:
- Netzentgelte: {refund_grid_fees:.2f} EUR
- Konzessionsabgaben: {refund_conc_fees:.2f} EUR
- Umlagen: {refund_levies:.2f} EUR
- Stromsteuer: {refund_el_tax:.2f} EUR
- Korrektur MwSt: {refund_vat:.2f} EUR
----------------------------------------
Summe Rückerstattung: {total_refund + refund_vat:.2f} EUR

Kosten mit Batterie (Netto nach Erstattung): {cost_net_after_refund:.2f} EUR
----------------------------------------
Gewinn durch Batteriesystem: {cost_base - cost_net_after_refund:.2f} EUR
"""
            print(summary)
            
            with open(SUMMARY_EXPORT_FILE, 'w') as f:
                f.write(summary)
            print(f"Summary gespeichert in: {SUMMARY_EXPORT_FILE}")
            
            df_final.to_csv(LOG_EXPORT_FILE, sep=';', decimal=',')
