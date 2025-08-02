import streamlit as st
import pandas as pd
import json
import openai
import os

# GPT-Modell
MODEL = "gpt-4o"

# API-Key aus Environment Variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Kein OpenAI API-Key gefunden. Bitte in den Streamlit Secrets hinterlegen.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Streamlit UI
st.title("📦 KI-basierte Bestellvorschläge")
st.markdown("""
Lade eine Artikelliste hoch und gib deine Firmenrichtlinie ein. Die KI analysiert deine Daten und gibt pro Artikel eine empfohlene Bestellmenge und Handlungsanweisung aus.
""")

# Firmenrichtlinien
firm_policy = st.text_area("🧠 Firmenrichtlinie eingeben", value="""
Wir möchten im Juli noch eine ausreichende Auswahl an Sommerartikeln verfügbar haben.
Ab 15. August beginnt der Abverkauf.
Am 31. August soll der Lagerbestand möglichst gering sein.
Restposten sollen max. 5 % des Anfangsbestands betragen.
Bei schwacher Nachfrage soll der Abverkauf früher starten.
""")

# Datei-Upload
file = st.file_uploader("📄 Artikeldaten (CSV) hochladen", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📊 Eingabedaten")
    st.dataframe(df)

    if st.button("🔍 Analyse starten"):
        with st.spinner("GPT analysiert die Artikel..."):
            # Prompt vorbereiten
            artikel_liste = df.to_dict(orient="records")
            system_prompt = f"""
Du bist ein Warenwirtschaftsanalyst.
Nutze folgende Firmenrichtlinie für deine Empfehlungen:
{firm_policy}

Für jeden Artikel sollst du Folgendes zurückgeben:
- \"article\": Name des Artikels
- \"order_quantity\": empfohlene Nachbestellmenge (ganzzahlig)
- \"action_recommendation\": Freitext-Vorschlag (z. B. Rabattieren, Abverkaufen, Preis halten)
- \"rationale\": Begründung in 1-2 Sätzen
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
                result = json.loads(content)
                out_df = pd.DataFrame(result)

                st.subheader("✅ Ergebnis")
                st.dataframe(out_df)

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Ergebnis als CSV herunterladen", csv, "bestellvorschlaege.csv")

            except Exception as e:
                st.error(f"Fehler bei der GPT-Verarbeitung: {e}")
