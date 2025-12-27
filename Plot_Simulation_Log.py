import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- KONFIGURATION ---

# Datei-Parameter
LOG_FILE_PATH = 'simulation_detailed_log_modul3_3Buck_e8deg_OktDezEXAA.csv'

# Technische Parameter (für Visualisierung von Limits/Targets)
BATTERY_CAPACITY_KWH = 71.7
SOC_TARGET_PERCENT = 0.40

# Ansichts-Steuerung
# Wähle hier, welchen Zeitraum du sehen willst.
# Wenn None, wird die Mitte des Datensatzes gewählt.
START_DATE_STR = '2025-11-22' # Format: '2024-12-10' oder None
VIEW_DAYS = 8         # Anzahl der Tage, die angezeigt werden sollen

def load_simulation_log(filepath):
    print(f"Lade Log-Datei: {filepath} ...")
    try:
        # Wichtig: Das Hauptskript speichert mit sep=';' und decimal=','
        df = pd.read_csv(filepath, sep=';', decimal=',', index_col=0)
        
        # Index zu Datetime konvertieren
        df.index = pd.to_datetime(df.index, utc=True)
        # Zeitzone entfernen oder anpassen für Plotting (damit x-Achse schön ist)
        df.index = df.index.tz_convert('Europe/Berlin')
        
        print(f"Daten geladen: {df.index.min()} bis {df.index.max()}")
        print(f"Spalten: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return None

def plot_results(df):
    if df is None or df.empty:
        print("Keine Daten zum Plotten.")
        return

    # Zeitfenster auswählen
    if START_DATE_STR:
        start_ts = pd.Timestamp(START_DATE_STR).tz_localize('Europe/Berlin')
    else:
        # Standard: Irgendwo in der Mitte starten
        mid_idx = len(df) // 2
        start_ts = df.index[mid_idx]
    
    end_ts = start_ts + pd.Timedelta(days=VIEW_DAYS)
    
    # Slice erstellen
    slice_df = df.loc[start_ts:end_ts]
    
    if slice_df.empty:
        print(f"Warnung: Kein Daten im Zeitraum {start_ts} bis {end_ts} gefunden.")
        # Fallback auf alles
        slice_df = df.iloc[:24*4*3] # Erste 3 Tage
    
    print(f"Plotte Zeitraum: {slice_df.index.min()} bis {slice_df.index.max()}")

    # --- PLOTTING ---
    # Layout: 3 Zeilen, unterste Zeile doppelt so hoch (height_ratios)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 16), 
                           gridspec_kw={'height_ratios': [1, 1, 2]})
    
    # ---------------------------------------------------------
    # SUBPLOT 1: PREISE
    # ---------------------------------------------------------
    ax[0].set_title("Strompreise")
    ax[0].set_ylabel("Preis [ct/kWh]")
    
    # Preise sind im CSV meist in EUR/kWh -> *100 für Cent
    if 'Price_Buy_Full' in slice_df.columns:
        ax[0].plot(slice_df.index, slice_df['Price_Buy_Full']*100, label='Preis (Bezug)', color='black', linestyle='--')
    if 'Price_Buy_Arb' in slice_df.columns:
        ax[0].plot(slice_df.index, slice_df['Price_Buy_Arb']*100, label='Arbitragebezug-Tarif (Bezug)', color='green')
    if 'DayAhead_EUR_kWh' in slice_df.columns:
        ax[0].plot(slice_df.index, slice_df['DayAhead_EUR_kWh']*100, label='Spotpreis (Verkauf)', color='red', alpha=0.4)
        
    ax[0].legend(loc='upper right', frameon=True)
    ax[0].grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # SUBPLOT 2: SOC (State of Charge)
    # ---------------------------------------------------------
    ax[1].set_title(f"Speicher Füllstand (Target SoC: {SOC_TARGET_PERCENT*100:.0f}%)")
    ax[1].set_ylabel("Energie [kWh]")
    
    if 'Opt_SoC_End_kWh' in slice_df.columns:
        # Gesamtkurve
        ax[1].plot(slice_df.index, slice_df['Opt_SoC_End_kWh'], color='black', label='Total', linewidth=1.5)
        
        # Gestapelte Flächen für Grün (PV) und Grau (Netz), falls vorhanden
        if 'Opt_SoC_End_Grey_Load' in slice_df.columns and 'Opt_SoC_End_Grey_Arb' in slice_df.columns and 'Opt_SoC_End_Green' in slice_df.columns:
            ax[1].stackplot(slice_df.index, 
                            slice_df['Opt_SoC_End_Grey_Load'], 
                            slice_df['Opt_SoC_End_Grey_Arb'], 
                            slice_df['Opt_SoC_End_Green'],
                            labels=['Grau (Hausnetz)','Orange (Arbitrage)','Grün (PV)'], 
                            colors=['grey','orange','lightgreen'],
                            alpha=0.6)
    
    # Target Linie und Kapazitätsgrenze
    ax[1].axhline(y=BATTERY_CAPACITY_KWH * SOC_TARGET_PERCENT, color='orange', linestyle=':', linewidth=2, label='Target SoC')
    ax[1].axhline(y=BATTERY_CAPACITY_KWH, color='black', linestyle='-', linewidth=0.8)
    
    ax[1].legend(loc='upper left', frameon=True)
    ax[1].grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # SUBPLOT 3: LEISTUNGSBILANZ (Kombiniert)
    # ---------------------------------------------------------
    ax[2].set_title("Leistungsbilanz (Positiv: Erzeugung/Laden | Negativ: Last/Entladen)")
    ax[2].set_ylabel("Leistung [kW]")
    
    # A) LINIEN PLOTS (Hintergrund, glatt)
    if 'PV_kW' in slice_df.columns:
        ax[2].plot(slice_df.index, slice_df['PV_kW'], label='PV Erzeugung', color='orange', alpha=0.8, linewidth=1.5)
    if 'Load_kW' in slice_df.columns:
        ax[2].plot(slice_df.index, -slice_df['Load_kW'], label='Hauslast', color='black', alpha=0.8, linewidth=1.5)
    
    # B) BALKEN PLOTS (Gestapelt)
    # Pandas Plotting berechnet Width oft in Tagen. 15min = 0.0104 Tage.
    width = 0.01 
    
    # 1. POSITIV: LADEN (PV + Netz)
    if 'Opt_PV2Batt_kW' in slice_df.columns and 'Opt_Grid2Batt_kW' in slice_df.columns:
        p2b = slice_df['Opt_PV2Batt_kW']
        g2b = slice_df['Opt_Grid2Batt_kW']
        
        ax[2].bar(slice_df.index, p2b, width=width, label='Laden (PV)', color='gold', align='center', alpha=0.7)
        ax[2].bar(slice_df.index, g2b, bottom=p2b, width=width, label='Laden (Netz)', color='purple', align='center', alpha=0.7)

    # 2. NEGATIV: ENTLADEN & BYPASS (Gestapelt nach unten)
    # Reihenfolge (von 0 nach unten): Grid2Load -> PV2Load -> Batt2Load -> Batt2Grid -> PV2Grid
    # Das gibt ein schönes Schichtbild.
    
    # Daten holen (und Vorzeichen umdrehen für den Plot)
    g2l = -slice_df.get('Opt_Grid2Load_kW', 0)
    pv2l = -slice_df.get('Opt_PV2Load_kW', 0)
    b2l = -slice_df.get('Opt_Batt2Load_kW', 0)
    b2g = -slice_df.get('Opt_Batt2Grid_kW', 0)
    pv2g = -slice_df.get('Opt_PV2Grid_kW', 0) # Falls vorhanden
    
    # Stapeln
    # 1. Netzbezug Haus (Grau)
    ax[2].bar(slice_df.index, g2l, width=width, label='Last (Netz)', color='darkgrey', align='center', alpha=0.7)
    
    # 2. PV Direktverbrauch (Orange)
    ax[2].bar(slice_df.index, pv2l, bottom=g2l, width=width, label='Last (PV)', color='orange', align='center', alpha=0.5)
    
    # 3. Batterie Entladung Haus (Blau)
    ax[2].bar(slice_df.index, b2l, bottom=g2l+pv2l, width=width, label='Entladen (Haus)', color='blue', align='center', alpha=0.7)
    
    # 4. Batterie Entladung Netz/Arbitrage (Rot)
    ax[2].bar(slice_df.index, b2g, bottom=g2l+pv2l+b2l, width=width, label='Entladen (Netz)', color='red', align='center', alpha=0.7)
    
    # 5. PV Einspeisung (Grün) - optional, falls im CSV
    if isinstance(pv2g, pd.Series) and pv2g.sum() < 0:
         ax[2].bar(slice_df.index, pv2g, bottom=g2l+pv2l+b2l+b2g, width=width, label='Einspeisung (PV)', color='lightgreen', align='center', alpha=0.5)

    # Nulllinie
    ax[2].axhline(0, color='black', linewidth=1)
    
    # Legende (2 Spalten)
    ax[2].legend(loc='lower left', ncol=3, fontsize='small', frameon=True)
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_log = load_simulation_log(LOG_FILE_PATH)
    if df_log is not None:
        plot_results(df_log)