import streamlit as st
import pandas as pd
import json
import openai
import os
from prophet import Prophet
import matplotlib.pyplot as plt
import datetime
import ast

# GPT-Modell
MODEL = "gpt-4o"

# API-Key aus Environment Variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Kein OpenAI API-Key gefunden. Bitte in den Streamlit Secrets hinterlegen.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Streamlit UI
st.title("KI-basierte Bestellvorschlaege und Absatzprognose")
st.markdown("""
Lade eine Artikelliste hoch und gib deine Firmenrichtlinie ein. Die KI analysiert deine Daten und gibt pro Artikel eine empfohlene Bestellmenge und Handlungsanweisung aus.
""")

# Firmenrichtlinie
firm_policy = st.text_area("Firmenrichtlinie eingeben", value="""
Wir möchten im Juli noch eine ausreichende Auswahl an Sommerartikeln verfügbar haben.
Ab 15. August beginnt der Abverkauf.
Am 31. August soll der Lagerbestand möglichst gering sein.
Restposten sollen max. 5 % des Anfangsbestands betragen.
Bei schwacher Nachfrage soll der Abverkauf früher starten.
""")

# Standort (als zusätzlicher Kontext für GPT)
location = st.text_input("Standort (für GPT-Kontext, z. B. 'Österreich')", value="Österreich")

# Rabattstrategie-Eingaben
st.subheader("Rabattstrategie definieren")
abverkaufsbeginn = st.date_input("Abverkaufsbeginn", value=datetime.date.today())
rabatt_phase_1 = st.number_input("Rabatt in Phase 1 (%)", value=30)
saisonende = st.date_input("Saisonende", value=datetime.date.today() + datetime.timedelta(days=30))
restwert_prozent = st.number_input("Restwert (% vom Einkaufspreis)", value=30)

# Datei-Upload
file = st.file_uploader("Artikeldaten (CSV) hochladen", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Eingabedaten")
    st.dataframe(df)

    if not {'einkaufspreis', 'verkaufspreis'}.issubset(df.columns):
        st.error("❌ Die CSV-Datei muss die Spalten 'einkaufspreis' und 'verkaufspreis' enthalten.")
        st.stop()

    # Forecasts vorbereiten
    st.subheader("Absatzprognose je Artikel anzeigen")
    forecasts = {}
    plots = {}
    artikelgruppen = df.groupby("artikel")
    selected_artikel = st.selectbox("Artikel auswählen", df['artikel'].unique())

    for artikel, gruppe in artikelgruppen:
        try:
            forecast_df = gruppe[["datum", "verkaufsmenge"]].rename(columns={"datum": "ds", "verkaufsmenge": "y"})
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

            modell = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="multiplicative"
            )
            modell.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            modell.fit(forecast_df)
            future = modell.make_future_dataframe(periods=6, freq='W')
            forecast = modell.predict(future)

            fig = modell.plot(forecast)
            plots[artikel] = fig

            forecast_values = forecast[['ds', 'yhat']].tail(6)
            forecasts[artikel] = {
                str(k.date()): int(v) for k, v in forecast_values.set_index('ds')['yhat'].items()
            }

        except Exception as e:
            forecasts[artikel] = {"error": str(e)}

    if selected_artikel in plots:
        st.write(f"Absatzprognose für: {selected_artikel}")
        st.pyplot(plots[selected_artikel])

    if st.button("Analyse starten"):
        with st.spinner("GPT analysiert die Artikel inklusive Prognosen..."):
            artikel_liste = df.groupby("artikel").first().reset_index().to_dict(orient="records")

            for artikel in artikel_liste:
                artikelname = artikel['artikel']
                artikel['forecast'] = forecasts.get(artikelname, {})

            system_prompt = f"""
Du bist ein Warenwirtschaftsanalyst mit Standort {location}.
Nutze folgende Firmenrichtlinie für deine Empfehlungen:
{firm_policy}

Die Rabattstrategie lautet:
- Abverkaufsbeginn: {abverkaufsbeginn.strftime('%d.%m.%Y')} mit {rabatt_phase_1}% Rabatt
- Saisonende: {saisonende.strftime('%d.%m.%Y')} mit Restwert von {restwert_prozent}% des Einkaufspreises

Du bekommst zu jedem Artikel eine Absatzprognose für die kommenden Wochen sowie Einkaufspreis und Verkaufspreis.

Für jeden Artikel sollst du Folgendes zurückgeben:
- "article": Name des Artikels
- "order_quantity": empfohlene Nachbestellmenge (ganzzahlig)
- "action_recommendation": Freitext-Vorschlag (z. B. Rabattieren, Abverkaufen, Preis halten)
- "rationale": Begründung in 1-2 Sätzen
- "scenario_comparison": JSON-Objekt mit Umsatz und Gewinn beider Strategien, z. B.: {{"ohne Rabatt": {{"Umsatz": 1000, "Gewinn": 500}}, "mit Rabatt": {{"Umsatz": 1200, "Gewinn": 600}}}}

Antworte ausschließlich mit einem JSON-Array, ohne Einleitung oder Kommentare.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hier sind die Artikeldaten mit Forecasts, Einkaufspreis und Verkaufspreis:\n{json.dumps(artikel_liste)}"}
            ]

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.2
                )
                content = response.choices[0].message.content

                try:
                    json_str = content[content.find("[") : content.rfind("]")+1]
                    result = json.loads(json_str)
                except Exception as e:
                    st.error(f"Fehler beim JSON-Parsing: {e}")
                    st.text(content)
                    st.stop()

                out_df = pd.DataFrame(result)

                st.subheader("Ergebnis")
                st.dataframe(out_df)

                # Szenarien-Vergleich visualisieren, wenn vorhanden
                if "scenario_comparison" in out_df.columns:
                    st.subheader("Szenarienvergleich (Umsatz/Gewinn)")
                    for i, row in out_df.iterrows():
                        try:
                            data = row["scenario_comparison"]
                            if isinstance(data, str):
                                data = json.loads(data)
                            df_comp = pd.DataFrame(data).T.reset_index()
                            df_comp.columns = ["Strategie", "Umsatz", "Gewinn"]

                            # Beste Strategie bestimmen
                            beste_strategie = df_comp.loc[df_comp['Gewinn'].idxmax(), 'Strategie']
                            row_colors = ["#d4edda" if strategie == beste_strategie else "#ffffff" for strategie in df_comp['Strategie']]

                            st.write(f"🧾 Artikel: **{row['article']}**")
                            st.table(df_comp.style.apply(lambda x: [
                                f"background-color: {color}" for color in row_colors], axis=1))

                            # Balkendiagramm
                            fig, ax = plt.subplots()
                            df_comp.plot(kind='bar', x='Strategie', y=['Umsatz', 'Gewinn'], ax=ax)
                            ax.set_ylabel("Wert (€)")
                            ax.set_title(f"Szenarienvergleich: {row['article']}")
                            st.pyplot(fig)

                        except Exception as e:
                            st.warning(f"Konnte Szenario für Artikel {row['article']} nicht visualisieren: {e}")

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Ergebnis als CSV herunterladen", csv, "bestellvorschlaege.csv")

            except Exception as e:
                st.error(f"Fehler bei der GPT-Verarbeitung: {e}")
