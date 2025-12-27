# Simulation der Wirtschaftlichkeit von Heimspeichern (MiSpeL / § 14a EnWG)

Dieses Repository enthält ein Python-Framework zur Simulation und wirtschaftlichen Optimierung von Heimspeichersystemen unter Berücksichtigung der aktuellen deutschen Gesetzgebung (Stand 2025), insbesondere der Neuregelungen durch das Solarpaket I und EnWG-Novellen.

## Funktionen

*   **Optimierung:** Einsatz eines MIP-Solvers (Mixed Integer Programming) via `cvxpy` zur Erstellung optimaler Fahrpläne für Laden und Entladen.
*   **Dynamische Tarife:** Berücksichtigung von Day-Ahead-Börsenstrompreisen (z.B. Tibber, aWATTar, EXAA).
*   **§ 14a EnWG (Modul 3):** Abbildung variabler Netzentgelte mit zeitabhängigen Tarifstufen (Niedriglast/Hochlast).
*   **MiSpeL (§ 118 Abs. 6 EnWG):** Detaillierte Simulation der Netzentgeltbefreiung für Speicherstrom. Das Modell berechnet ex-post die erstattungsfähigen Mengen basierend auf dem physikalischen und regulatorischen Stromfluss (Trennung von Netz- und PV-Strom im Speicher).
*   **Degradation:** Berücksichtigung von zyklischer Alterung durch Kostenstrafen im Optimierungsmodell.
*   **Multi-Bucket-Modell:** Virtuelle Trennung des Speicherinhalts in "Graustrom" (Netzbezug für Eigenverbrauch), "Arbitrage-Strom" (Netzbezug für Rückspeisung) und "Grünstrom" (PV).

## Voraussetzungen

*   Python 3.8 oder höher
*   Die folgenden Python-Bibliotheken:
    *   `pandas`
    *   `numpy`
    *   `cvxpy`
    *   `matplotlib` (für die Visualisierung)

### Solver Hinweis
Das Skript ist standardmäßig für den Solver **SCIP** konfiguriert (`solver=cp.SCIP`). Da es sich um ein gemischt-ganzzahliges Problem (MIP) handelt, wird ein entsprechender Solver benötigt. Alternativen wie **CBC**, **GLPK** oder kommerzielle Solver wie **Gurobi** können ebenfalls verwendet werden, erfordern aber ggf. Anpassungen im Code (`problem.solve(solver=...)`) und entsprechende Installationen.

## Installation

1.  Repository klonen:
    ```bash
    git clone https://github.com/dein-username/dein-repo-name.git
    cd dein-repo-name
    ```

2.  Abhängigkeiten installieren:
    ```bash
    pip install pandas numpy "cvxpy[SCIP]" matplotlib
    ```

## Nutzung

1.  **Daten vorbereiten:**
    Lege deine Preisdaten (z.B. `Day Ahead...csv`) und PV-Erzeugungsdaten (`...pv_daten...csv`) im Verzeichnis ab. Die Formate müssen den Erwartungen im Skript entsprechen (siehe `load_data` Funktion in `Neu_MIP_solver_EnWG_e8_Deg_Split_Mod3.py`).

2.  **Konfiguration:**
    Öffne das Hauptskript `Neu_MIP_solver_EnWG_e8_Deg_Split_Mod3.py` und passe die Parameter am Anfang der Datei an:
    *   Dateipfade (`PRICE_FILE_PATH`, `PV_DATA_FILE_PATH`)
    *   Batteriegröße (`BATTERY_CAPACITY_KWH`, `BATTERY_POWER_KW`)
    *   Kostenparameter (Umlagen, Steuern)

3.  **Simulation starten:**
    ```bash
    python Neu_MIP_solver_EnWG_e8_Deg_Split_Mod3.py
    ```
    Das Skript erstellt eine Log-Datei (`.csv`) und eine Zusammenfassung (`.txt`).

4.  **Ergebnisse visualisieren:**
    Nutze das Plotting-Skript, um die Fahrpläne grafisch darzustellen:
    ```bash
    python Plot_Simulation_Log.py
    ```
    *(Stelle sicher, dass der Dateiname im Plot-Skript mit dem Output der Simulation übereinstimmt)*

## Lizenz

Dieses Projekt ist unter der **Mozilla Public License 2.0 (MPL 2.0)** lizenziert. Siehe LICENSE Datei für Details.

Copyright 2025 Lukas Neusius

## Haftungsausschluss

Diese Software dient ausschließlich zu Simulations- und Bildungszwecken. Die Berechnungen stellen keine Finanzberatung dar. Trotz sorgfältiger Programmierung können Fehler enthalten sein. Die Anwendung auf reale wirtschaftliche Entscheidungen erfolgt auf eigene Gefahr.
