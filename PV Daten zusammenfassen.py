import pandas as pd
import glob
import os

def merge_csv_files():
    # Name des Unterordners
    subfolder = "PV Daten wochen"
    
    # Prüfen, ob der Ordner existiert
    if not os.path.exists(subfolder):
        print(f"Fehler: Der Ordner '{subfolder}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass der Ordner existiert und richtig geschrieben ist.")
        return

    # Suche nach allen CSV-Dateien im Unterordner
    # os.path.join sorgt dafür, dass der Pfad auf Windows und Mac/Linux funktioniert
    search_path = os.path.join(subfolder, "*.csv")
    csv_files = glob.glob(search_path)
    
    # Liste, um die Daten aller Dateien zu sammeln
    all_data_frames = []
    
    print(f"{len(csv_files)} Dateien in '{subfolder}' gefunden. Beginne Verarbeitung...")

    for filename in csv_files:
        # Wir prüfen nur den Dateinamen selbst, falls die Ausgabedatei versehentlich dort landet
        if os.path.basename(filename) == "zusammengefasste_pv_daten_komplett.csv":
            continue

        try:
            # 1. Datei einlesen
            # na_values=['-']: WICHTIG! Behandelt "-" als fehlenden Wert (NaN). 
            # Das verhindert, dass die Spalte als "Text" erkannt wird, wodurch das Komma-Problem gelöst wird.
            df = pd.read_csv(filename, sep=';', skiprows=11, decimal=',', na_values=['-'], encoding='utf-8')
            
            # Spalten bereinigen (Leerzeichen entfernen etc.)
            df.columns = df.columns.str.strip()

            # Prüfen, ob die erwarteten Spalten vorhanden sind
            if "Zeitraum" in df.columns and "Leistung [kW]" in df.columns:
                # Eventuelle NaNs (die vorher "-" waren) mit 0 füllen
                df['Leistung [kW]'] = df['Leistung [kW]'].fillna(0)
                all_data_frames.append(df)
            else:
                print(f"Warnung: Datei '{filename}' hat nicht das erwartete Format und wird übersprungen.")

        except Exception as e:
            print(f"Fehler beim Lesen von {filename}: {e}")

    # Wenn keine Daten gefunden wurden, abbrechen
    if not all_data_frames:
        print("Keine gültigen Daten gefunden.")
        return

    # 2. Alle Tabellen zu einer einzigen zusammenfügen
    combined_df = pd.concat(all_data_frames, ignore_index=True)

    # 3. Datumsformat bereinigen und Zeitzone hinzufügen
    # Die Zeitstempel werden als naive Zeiten (ohne Zeitzone) eingelesen.
    naive_timestamps = pd.to_datetime(combined_df['Zeitraum'], dayfirst=True)
    # Anschließend wird die korrekte Zeitzone 'Europe/Berlin' zugewiesen.
    # tz_localize kümmert sich automatisch um die korrekte Sommer-/Winterzeit.
    combined_df['Zeitraum'] = naive_timestamps.dt.tz_localize('Europe/Berlin')

    # 4. Duplikate entfernen (falls sich Dateien überschneiden)
    combined_df = combined_df.drop_duplicates(subset=['Zeitraum'])

    # 5. Lücken füllen (Resampling)
    # Wir setzen den Zeitraum als Index, um Zeit-Operationen durchzuführen
    combined_df = combined_df.set_index('Zeitraum')
    
    # '15min' erstellt ein Raster alle 15 Minuten vom allerersten bis zum allerletzten Datum
    # .asfreq() erstellt die fehlenden Zeilen
    # .fillna(0) füllt die neuen leeren Felder (Nachtstunden) mit 0
    combined_df = combined_df.resample('15min').asfreq().fillna(0)

    # Index wieder zur normalen Spalte machen
    combined_df = combined_df.reset_index()

    # 6. Speichern im neuen Format (im Hauptordner, nicht im Unterordner)
    output_filename = "zusammengefasste_pv_daten_komplett.csv"
    
    # float_format='%.2f': Erzwingt 2 Nachkommastellen (z.B. 0.00 statt 0)
    # Durch das Entfernen von 'date_format' schreibt Pandas automatisch das ISO 8601 Format mit Zeitzoneninformation.
    combined_df.to_csv(output_filename, index=False, sep=',', decimal='.', float_format='%.2f')

    print(f"Erfolg! Alle Daten wurden in '{output_filename}' gespeichert.")
    print(f"Format: Komma-getrennt, Dezimal-Punkt, ISO-Datum mit Zeitzone.")
    print(f"Zeitraum: {combined_df['Zeitraum'].min()} bis {combined_df['Zeitraum'].max()}")
    print(f"'-' Zeichen wurden erfolgreich zu 0.00 umgewandelt.")

if __name__ == "__main__":
    merge_csv_files()