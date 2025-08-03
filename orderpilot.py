import streamlit as st
import pandas as pd
import json
import openai
import os
from prophet import Prophet
import matplotlib.pyplot as plt

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

# Datei-Upload
file = st.file_uploader("Artikeldaten (CSV) hochladen", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Eingabedaten")
    st.dataframe(df)

    # Forecasts vorbereiten
    st.subheader("Absatzprognose je Artikel anzeigen")
    selected_artikel = st.selectbox("Artikel auswählen", df['artikel'].unique())
    forecasts = {}
    artikelgruppen = df.groupby("artikel")
    for artikel, gruppe in artikelgruppen:
        try:
            forecast_df = gruppe[["datum", "verkaufsmenge"]].rename(columns={"datum": "ds", "verkaufsmenge": "y"})
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

            modell = Prophet()
            modell.fit(forecast_df)
            future = modell.make_future_dataframe(periods=6, freq='W')
            forecast = modell.predict(future)

            if artikel == selected_artikel:
                st.write(f"Absatzprognose für: {artikel}")
                fig = modell.plot(forecast)
                st.pyplot(fig)

            # Nur relevante Prognose-Werte extrahieren (nächste 6 Wochen)
            forecast_values = forecast[['ds', 'yhat']].tail(6)
            forecasts[artikel] = {
    str(k.date()): int(v) for k, v in forecast_values.set_index('ds')['yhat'].items()
}

        except Exception as e:
            forecasts[artikel] = {"error": str(e)}

    if st.button("Analyse starten"):
        with st.spinner("GPT analysiert die Artikel inklusive Prognosen..."):
            artikel_liste = df.groupby("artikel").first().reset_index().to_dict(orient="records")

            # Forecasts an Artikeldaten anhängen
            for artikel in artikel_liste:
                artikelname = artikel['artikel']
                artikel['forecast'] = forecasts.get(artikelname, {})

            system_prompt = f"""
Du bist ein Warenwirtschaftsanalyst mit Standort {location}.
Nutze folgende Firmenrichtlinie für deine Empfehlungen:
{firm_policy}

Du bekommst zu jedem Artikel eine Absatzprognose für die kommenden Wochen. Berücksichtige dabei den Standort und typische Saisonalität (z. B. Sommerartikel).

Für jeden Artikel sollst du Folgendes zurückgeben:
- "article": Name des Artikels
- "order_quantity": empfohlene Nachbestellmenge (ganzzahlig)
- "action_recommendation": Freitext-Vorschlag (z. B. Rabattieren, Abverkaufen, Preis halten)
- "rationale": Begründung in 1-2 Sätzen

Antworte ausschließlich mit einem JSON-Array, ohne Einleitung oder Kommentare.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hier sind die Artikeldaten mit Forecasts:\n{json.dumps(artikel_liste)}"}
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

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Ergebnis als CSV herunterladen", csv, "bestellvorschlaege.csv")

            except Exception as e:
                st.error(f"Fehler bei der GPT-Verarbeitung: {e}")
