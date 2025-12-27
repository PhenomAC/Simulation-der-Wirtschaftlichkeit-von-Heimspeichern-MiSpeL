# Simulation der Wirtschaftlichkeit von Heimspeichern (MiSpeL / § 118 EnWG)

Dieses Repository enthält ein Python-Simulationsskript zur Analyse der Profitabilität eines AC-gekoppelten Heimspeichers in Kombination mit einer PV-Anlage.

Der Fokus liegt auf der **Marktintegration von Speichern** unter den neuen regulatorischen Rahmenbedingungen in Deutschland (MiSpeL, EnWG-Novelle), die einen Mischbetrieb aus Eigenverbrauchsoptimierung, optimierter Einspeisung mit Direktvermarktung und Arbitrage (Handel mit Netzstrom) wirtschaftlich attraktiv machen.
Aktuell geht die Simulation davon aus, dass für die saldierungsfähige Netzeinspeisung keine weiteren Kosten außer den Day-Ahead-Einkaufskosten und einem kleinen Anteil an festen Beschaffungskosten anfallen. Das ist die optimale Annahme. Bei den Konzessionsabgaben und den diversen Steuern ist die Befreiung aber noch nicht explizit geklärt. Das bedarf noch weiteren Klarstellungen seitens des Gesetzgebers.

## Regulatorischer Hintergrund

Die Simulation modelliert die Auswirkungen der Neuregelungen zur Marktintegration von Speichern und Ladepunkten (**MiSpeL**) sowie der Novellierung des **§ 118 Abs. 6 EnWG**.

### Das Problem: "Ausschließlichkeit" (Alte)
Bisher mussten Betreiber wählen:
*   **Reiner EE-Speicher:** Nur PV-Strom laden (EEG-Vergütung möglich, aber kein Laden aus dem Netz erlaubt).
*   **Reiner Netz-Speicher:** Nur Netzstrom laden (Netzentgeltbefreiung möglich, aber keine EEG-Vergütung für PV-Strom).

Ein Mischbetrieb führte oft zum Verlust der Privilegien.

### Die Lösung: Abgrenzungsoption (Neue)
Durch die neuen Regelungen wird ein Mischbetrieb ermöglicht. Die Strommengen werden nicht mehr physikalisch getrennt, sondern **rechnerisch abgegrenzt** (siehe https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/ErneuerbareEnergien/EEG_Aufsicht/MiSpeL/start.html, Fallkonstellation A1 der MiSpeL-Eckpunkte).

1.  **Saldierungsfähige Netzeinspeisung:** Es wird rechnerisch ermittelt, welcher Anteil des Stroms im Speicher aus dem Netz stammt. Wird dieser wieder eingespeist (Arbitrage), werden die darauf gezahlten **Umlagen, Stromsteuer und Netzentgelte zurückerstattet** (bzw. saldiert).
2.  **Anteilige Netzentgeltbefreiung (§ 118 Abs. 6 EnWG):** Die Befreiung von Netzentgelten gilt nun auch anteilig für den wieder eingespeisten Netzstrom. Dies macht Arbitrage-Geschäfte (Laden bei niedrigen Preisen/Niedriglasttarif, Entladen zu Hochpreiszeiten) für Heimspeicher erst interessant.
3.  **Gewillkürte Vorrangregelung:** Bei Gleichzeitigkeit von Last und Ladung bzw. Einspeisung und Entladung gelten gesetzlich definierte Vorrangregeln, die in der Simulation berücksichtigt werden (z.B. gilt Speicherladung bei gleichzeitigem Netzbezug vorrangig als Netzladung).

---

## Funktionsweise der Simulation

Das Skript nutzt mathematische Optimierung, um den idealen Fahrplan für den Speicher zu berechnen.

### 1. Optimierungsmodell (MIP Solver)
Es wird ein **Mixed-Integer Programming (MIP)** Ansatz verwendet (via `cvxpy` und `SCIP` Solver). Das Modell entscheidet für jedes 15-Minuten-Intervall:
*   Soll geladen oder entladen werden? (Binäre Entscheidung zur Vermeidung von gleichzeitigem Laden/Entladen).
*   Wieviel Strom fließt in welchen "Topf"?

### 2. Das 3-Bucket-Modell
Um die Kosten und regulatorischen Kategorien korrekt zuzuordnen, unterteilt die Simulation den Speicher virtuell in drei Bereiche ("Buckets"):
*   🟢 **Green Bucket:** PV-Strom. Kostenlos. Vorrangig für Eigenverbrauch, Überschuss für EEG-Einspeisung.
*   ⚪ **Grey Load Bucket:** Netzstrom zum vollen Preis (inkl. Abgaben). Bestimmt für den zeitversetzten Eigenverbrauch (z.B. um Hochpreisphasen zu brücken).
*   🟠 **Grey Arbitrage Bucket:** Netzstrom zu Grenzkosten (Spotpreis + nicht-erstattungsfähige Gebühren). **Darf nur zurück ins Netz entladen werden.**

### 3. Kostenstruktur
*   **Day-Ahead Preise:** Stündlich variable Börsenstrompreise.
*   **Variable Netzentgelte (§ 14a EnWG Modul 3):** Zeitabhängige Netzentgelte (Niedriglast-, Standard-, Hochlastfenster).
*   **Rückerstattung:** Ex-Post-Berechnung der erstattungsfähigen Entgelte gemäß MiSpeL-Formeln.

---

## Installation & Nutzung

### Voraussetzungen
*   Python 3.x
*   Solver: **SCIP** (oder ein anderer MIP-fähiger Solver wie Gurobi/CPLEX) muss installiert sein.
*   Bibliotheken:
    ```bash
    pip install pandas numpy "cvxpy[SCIP]" matplotlib
    ```

### Ausführung
*   Pfade zu den CSV-Dateien (Strompreise, PV-Daten) im Skript anpassen.
*   Die Strompreise kann man sich von energy-charts.info herunterladen. Die Simulation arbeitet in 15 Minuten Intervallen. Also muss man auch die neuen 15 minütigen Day-Ahead Preise als Basis nehmen. Da diese erst im Oktober eingeführt wurden, kann man auch die EXAA Daten für die gesamte Zeit davor nehmen. Der Unterschied ist gering und eine Stichprobe von Oktober-Dezember ergab keine signifikanten Unterschiede im Simulationsergebnis.
*   Die PV Daten stammen vorzugsweise von der eigenen PV-Anlage. Ansonsten kann man sich Daten von PVGIS erzeugen lassen oder man nimmt die PV Ertragsdaten des eigenen Bundeslandes von energy-charts.info und skaliert diese auf einen sinnvollen Jahresertrag. Die Daten aus den Bundesländern sind natürlich sehr viel "glatter" als die Daten einer realen PV-Anlage mit Wolken die plötzlich Schatten erzeugen. Es werden Daten im ISO 8601 Format erwartet.
*   Simulation starten:
    ```bash
    Simulationsskript.py
    ```
*   Ergebnisse visualisieren:
    ```bash
    python Plot_Simulation_Log.py
    ```
## Disclaimer
Dieses Tool dient der privaten Abschätzung und Modellierung. Die regulatorischen Rahmenbedingungen sind komplex und teilweise noch in Konsultationsphasen. Es wird keine Gewähr für die Richtigkeit der steuerlichen und rechtlichen Berechnungen übernommen.
