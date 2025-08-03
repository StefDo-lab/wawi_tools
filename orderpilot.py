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
    st.error("\u26a0\ufe0f Kein OpenAI API-Key gefunden. Bitte in den Streamlit Secrets hinterlegen.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Streamlit UI
st.title("\ud83d\udce6 KI-basierte Bestellvorschl\u00e4ge + Absatzprognose")
st.markdown("""
Lade eine Artikelliste hoch und gib deine Firmenrichtlinie ein. Die KI analysiert deine Daten und gibt pro Artikel eine empfohlene Bestellmenge und Handlungsanweisung aus.
""")

# Firmenrichtlinie
firm_policy = st.text_area("\ud83e\udde0 Firmenrichtlinie eingeben", value="""
Wir m\u00f6chten im Juli noch eine ausreichende Auswahl an Sommerartikeln verf\u00fcgbar haben.
Ab 15. August beginnt der Abverkauf.
Am 31. August soll der Lagerbestand m\u00f6glichst gering sein.
Restposten sollen max. 5 % des Anfangsbestands betragen.
Bei schwacher Nachfrage soll der Abverkauf fr\u00fcher starten.
""")

# Datei-Upload
file = st.file_uploader("\ud83d\udcc4 Artikeldaten (CSV) hochladen", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("\ud83d\udcca Eingabedaten")
    st.dataframe(df)

    # Prophet-Prognose vorbereiten (vereinfachtes Beispiel f\u00fcr 1 Artikel)
    st.subheader("\ud83d\udcca Absatzprognose mit Prophet (Artikel 1)")
    try:
        forecast_df = df.copy()
        forecast_df = forecast_df.rename(columns={"datum": "ds", "verkaufsmenge": "y"})
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        modell = Prophet()
        modell.fit(forecast_df)
        future = modell.make_future_dataframe(periods=6, freq='W')
        forecast = modell.predict(future)

        fig = modell.plot(forecast)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Prognose konnte nicht erstellt werden: {e}")

    if st.button("\ud83d\udd0d Analyse starten"):
        with st.spinner("GPT analysiert die Artikel..."):
            # Prompt vorbereiten
            artikel_liste = df.to_dict(orient="records")
            system_prompt = f"""
Du bist ein Warenwirtschaftsanalyst.
Nutze folgende Firmenrichtlinie f\u00fcr deine Empfehlungen:
{firm_policy}

F\u00fcr jeden Artikel sollst du Folgendes zur\u00fcckgeben:
- \"article\": Name des Artikels
- \"order_quantity\": empfohlene Nachbestellmenge (ganzzahlig)
- \"action_recommendation\": Freitext-Vorschlag (z.\u202fB. Rabattieren, Abverkaufen, Preis halten)
- \"rationale\": Begr\u00fcndung in 1-2 S\u00e4tzen

Antworte ausschlie\u00dflich mit einem JSON-Array, ohne Einleitung oder Kommentare.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hier sind die Artikeldaten:\n{json.dumps(artikel_liste)}"}
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

                st.subheader("\u2705 Ergebnis")
                st.dataframe(out_df)

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("\ud83d\udcc5 Ergebnis als CSV herunterladen", csv, "bestellvorschlaege.csv")

            except Exception as e:
                st.error(f"Fehler bei der GPT-Verarbeitung: {e}")
