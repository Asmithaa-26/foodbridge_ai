import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FoodBridge AI", layout="wide")

st.title("FoodBridge AI Platform")
st.subheader("Intelligent Food Waste Redistribution System")

DATA_PATH = "data/food_master_dataset.csv"

df = pd.read_csv(DATA_PATH)

city_coords = {
    "Chennai": (13.0827,80.2707),
    "Bangalore": (12.9716,77.5946),
    "Hyderabad": (17.3850,78.4867),
    "Mumbai": (19.0760,72.8777),
    "Delhi": (28.7041,77.1025)
}

df["latitude"] = df["city"].map(lambda x: city_coords[x][0])
df["longitude"] = df["city"].map(lambda x: city_coords[x][1])

# Navigation
page = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard Analytics",
        "Food Waste Prediction",
        "FoodBridge AI Assistant",
        "Sentiment Analysis",
        "Language Translation",
        "NGO Route Optimization"
    ]
)

# ------------------------------------------------
# PAGE 1 : ANALYTICS DASHBOARD
# ------------------------------------------------

if page == "Dashboard Analytics":

    st.header("FoodBridge AI Analytics Dashboard")

    total_surplus = df["surplus_kg"].sum()
    total_prepared = df["total_prepared_kg"].sum()
    total_people = df["number_of_people"].sum()
    avg_surplus = df["surplus_kg"].mean()

    people_fed = int(total_surplus * 3)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Surplus (kg)", round(total_surplus,2))
    col2.metric("Total Prepared Food", round(total_prepared,2))
    col3.metric("People Served", int(total_people))
    col4.metric("Average Surplus", round(avg_surplus,2))
    col5.metric("Potential Meals", people_fed)

    # ------------------------------------------------
    # SURPLUS BY CITY
    # ------------------------------------------------

    st.subheader("Surplus Distribution by City")

    city_surplus = df.groupby("city")["surplus_kg"].sum().reset_index()

    fig1 = px.bar(city_surplus, x="city", y="surplus_kg", color="city")

    st.plotly_chart(fig1, theme="streamlit")

    # ------------------------------------------------
    # OUTLET TYPE CONTRIBUTION
    # ------------------------------------------------

    st.subheader("Outlet Type Contribution")

    outlet_data = df.groupby("outlet_type")["surplus_kg"].sum().reset_index()

    fig2 = px.pie(outlet_data, values="surplus_kg", names="outlet_type")

    st.plotly_chart(fig2, theme="streamlit")

    # ------------------------------------------------
    # WEATHER IMPACT
    # ------------------------------------------------

    st.subheader("Weather Impact on Surplus")

    weather_data = df.groupby("rainfall_category")["surplus_kg"].mean().reset_index()

    fig3 = px.bar(weather_data, x="rainfall_category", y="surplus_kg", color="rainfall_category")

    st.plotly_chart(fig3, theme="streamlit")

    # ------------------------------------------------
    # LIVE SURPLUS MAP
    # ------------------------------------------------

    st.subheader("Live Food Surplus Locations")

    fig_map = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    size="surplus_kg",
    color="surplus_kg",
    hover_name="city",
    hover_data=["outlet_type","surplus_kg"],
    zoom=4,
    mapbox_style="open-street-map"
)

    st.plotly_chart(fig_map, theme="streamlit")
    # ------------------------------------------------
    # DATASET VIEWER
    # ------------------------------------------------

    st.subheader("Dataset Explorer")

    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df)


# ------------------------------------------------
# PAGE 2 : FOOD WASTE PREDICTION
# ------------------------------------------------

elif page == "Food Waste Prediction":

    regressor = joblib.load("saved_models/foodbridge_regressor.pkl")
    classifier = joblib.load("saved_models/waste_classifier.pkl")

    with open("saved_models/model_features.json") as f:
        reg_features = json.load(f)

    with open("saved_models/classifier_features.json") as f:
        clf_features = json.load(f)

    def prepare_input(data,features):

        df_input=pd.DataFrame([data])
        df_input=pd.get_dummies(df_input)

        for col in features:
            if col not in df_input.columns:
                df_input[col]=0

        df_input=df_input[features]

        return df_input


    st.sidebar.header("Input Data")

    number_of_people=st.sidebar.slider("Number of People",10,1000,100)

    meals_per_person=st.sidebar.slider("Meals per Person",1.0,3.0,1.5)

    footfall_count=st.sidebar.slider("Footfall",10,1000,120)

    total_prepared_kg=st.sidebar.slider("Prepared Food (kg)",10,1000,200)

    rainfall_mm=st.sidebar.slider("Rainfall",0,100,5)

    temperature_c=st.sidebar.slider("Temperature",10,45,30)

    city=st.sidebar.selectbox("City",["Chennai","Bangalore","Hyderabad","Mumbai","Delhi"])

    district=st.sidebar.selectbox("District",
    ["Chennai","Bangalore Urban","Hyderabad","Mumbai Suburban","New Delhi"])

    outlet_type=st.sidebar.selectbox("Outlet Type",["restaurant","supermarket","farm"])

    rainfall_category=st.sidebar.selectbox("Rainfall Category",["low","medium","high"])

    temperature_category=st.sidebar.selectbox("Temperature Category",["cool","moderate","hot"])


    input_data={
        "number_of_people":number_of_people,
        "meals_per_person":meals_per_person,
        "footfall_count":footfall_count,
        "total_prepared_kg":total_prepared_kg,
        "rainfall_mm":rainfall_mm,
        "temperature_c":temperature_c,
        "city":city,
        "district":district,
        "outlet_type":outlet_type,
        "rainfall_category":rainfall_category,
        "temperature_category":temperature_category
    }

    if st.button("Predict Food Waste"):

        X_reg=prepare_input(input_data,reg_features)
        X_clf=prepare_input(input_data,clf_features)

        surplus=regressor.predict(X_reg)[0]
        waste_level=classifier.predict(X_clf)[0]

        st.success("Prediction Results")

        c1,c2=st.columns(2)

        c1.metric("Predicted Surplus (kg)",round(surplus,2))
        c2.metric("Waste Level",waste_level)


# ------------------------------------------------
# PAGE 3 : CHATBOT
# ------------------------------------------------

elif page == "FoodBridge AI Assistant":

    st.header("AI Volunteer Assistant")

    documents=[
    "Food surplus should be redistributed within six hours.",
    "Cooked food must be stored below five degree Celsius.",
    "NGOs collect surplus food from restaurants.",
    "Volunteers must inspect food quality before distribution.",
    "Transportation should minimize travel time to reduce waste."
    ]

    model=SentenceTransformer("all-MiniLM-L6-v2")

    doc_vectors=model.encode(documents)

    dim=doc_vectors.shape[1]

    index=faiss.IndexFlatL2(dim)

    index.add(np.array(doc_vectors))


    def chatbot(query):

        q_vec=model.encode([query])

        dist,ind=index.search(np.array(q_vec),k=2)

        context=" ".join([documents[i] for i in ind[0]])

        return context


    question=st.text_input("Ask a question")

    if st.button("Ask Assistant"):

        answer=chatbot(question)

        st.success("Response")

        st.write(answer)


# ------------------------------------------------
# PAGE 4 : SENTIMENT ANALYSIS
# ------------------------------------------------

elif page == "Sentiment Analysis":

    st.header("Feedback Sentiment Analysis")

    model=joblib.load("saved_models/sentiment_model.pkl")

    vectorizer=joblib.load("saved_models/tfidf_vectorizer.pkl")

    text=st.text_area("Enter feedback")

    if st.button("Analyze"):

        vec=vectorizer.transform([text])

        pred=model.predict(vec)[0]

        st.metric("Sentiment",pred)


# ------------------------------------------------
# PAGE 5 : TRANSLATION
# ------------------------------------------------

elif page == "Language Translation":

    st.header("Volunteer Language Translator")

    from transformers import MBartForConditionalGeneration,MBart50TokenizerFast

    model_name="facebook/mbart-large-50-many-to-many-mmt"

    tokenizer=MBart50TokenizerFast.from_pretrained(model_name)

    model=MBartForConditionalGeneration.from_pretrained(model_name)

    tokenizer.src_lang="en_XX"

    text=st.text_area("Enter English Text")

    if st.button("Translate"):

        encoded=tokenizer(text,return_tensors="pt")

        tokens=model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"]
        )

        translation=tokenizer.batch_decode(tokens,skip_special_tokens=True)

        st.success("Tamil Translation")

        st.write(translation[0])
        
# ------------------------------------------------
# PAGE 6 : ROUTE OPTIMIZATION
# ------------------------------------------------

elif page == "NGO Route Optimization":

    st.header("NGO Food Collection Route Optimization")

    st.write("This module finds the most efficient route for collecting surplus food from multiple locations.")

    locations = df.groupby("city")[["latitude","longitude"]].mean().reset_index()

    st.subheader("Available Collection Locations")

    st.dataframe(locations)

    import math

    def distance(coord1, coord2):
        return math.sqrt(
            (coord1[0]-coord2[0])**2 +
            (coord1[1]-coord2[1])**2
        )

    coords = locations[["latitude","longitude"]].values

    n = len(coords)

    dist_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = distance(coords[i],coords[j])

    visited = [False]*n

    route = [0]

    visited[0] = True

    for _ in range(n-1):

        last = route[-1]

        next_city = None

        min_dist = float("inf")

        for j in range(n):

            if not visited[j] and dist_matrix[last][j] < min_dist:

                min_dist = dist_matrix[last][j]

                next_city = j

        route.append(next_city)

        visited[next_city] = True

    route.append(0)

    route_cities = locations.iloc[route]["city"].tolist()

    st.subheader("Optimized NGO Collection Route")

    st.write(" → ".join(route_cities))

    route_df = locations.iloc[route]

    fig = px.line_mapbox(
        route_df,
        lat="latitude",
        lon="longitude",
        hover_name="city",
        zoom=4,
        mapbox_style="open-street-map"
    )

    st.plotly_chart(fig,theme="streamlit")

    st.success("Optimized route reduces travel distance and food spoilage risk.")