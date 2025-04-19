import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import streamlit as st

# Load model + embedder once
@st.cache_resource
def load_models():
    model = xgb.Booster()
    model.load_model("tm_lambdamart_ranking_model.json")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embedder

# Load dataset once
@st.cache_data
def load_data():
    df = pd.read_csv("ticketmaster.zip")
    df["text"] = df.apply(lambda row: f"{row['segmentname']} {row['genrename']} {row['eventname']}  {row['venuecity']}", axis=1)
    unique_texts = df["text"].unique()
    vector_map = {t: embedder.encode(t, convert_to_numpy=True) for t in unique_texts}
    df["vector"] = df["text"].map(vector_map)
    return df

# Convert query to vector
def text_to_vec(text):
    return embedder.encode(text, convert_to_numpy=True)

# Feature engineering
def city_match(row, query):
    return int(row["venuecity"].lower() in query.lower())

def segment_match_score(row, query):
    return sum(1 for word in query.lower().split() if word in str(row["segmentname"]).lower())

def genre_match_score(row, query):
    return sum(1 for word in query.lower().split() if word in str(row["genrename"]).lower())

def date_match(row, query):
    return int("today" in query.lower() and pd.to_datetime(row["eventdatetime"]).date() == pd.Timestamp.today().date())

def price_score(row):
    try:
        return 1 if 20 <= float(row["pricemax"]) <= 100 else 0
    except:
        return 0

# Ranking logic
def rank_query(query, model, event_df):
    query_vec = text_to_vec(query)

    event_df["similarity"] = event_df["vector"].apply(lambda vec: 1 - cosine(query_vec, vec))
    event_df["city_match"] = event_df.apply(lambda r: city_match(r, query), axis=1)
    event_df["segment_match_score"] = event_df.apply(lambda r: segment_match_score(r, query), axis=1)
    event_df["genre_match_score"] = event_df.apply(lambda r: genre_match_score(r, query), axis=1)
    event_df["date_match"] = event_df.apply(lambda r: date_match(r, query), axis=1)
    event_df["price_score"] = event_df.apply(lambda r: price_score(r), axis=1)

    features = ["similarity", "city_match", "segment_match_score", "genre_match_score", "date_match", "price_score"]
    dmatrix = xgb.DMatrix(event_df[features])
    event_df["predicted_score"] = model.predict(dmatrix)

    top_results = (
        event_df
        .sort_values(by="predicted_score", ascending=False)
        .drop_duplicates(subset="eventname")
        .head(5)
    )
    return top_results

# --- Streamlit UI ---
st.title("🎟️ Event Ranking Search")
st.markdown("Type in an event search query and get the top 5 ranked results.")

query = st.text_input("Enter your query:",  )

if query:
    model, embedder = load_models()
    event_df = load_data()
    with st.spinner("Ranking events..."):
        results = rank_query(query, model, event_df)

    st.subheader("🔝 Top 5 Results")
    for i, row in results.iterrows():
        if isinstance(row["primaryimageurl"], str) and row["primaryimageurl"].startswith("http"):
            st.image(row["primaryimageurl"], width=350, caption=row["eventname"])
        st.markdown(f"**{row['eventname']}** - {row['venuecity']}, {row['venuestate']}")
        st.markdown(f"- 🗓️ Date: {row['eventdate']}") 
        st.markdown(f"- 📍 Venue: {row['venuename']}")
        st.markdown(f"- 🌐 [View Event]({row['eventurl']})")
        st.markdown("---")
