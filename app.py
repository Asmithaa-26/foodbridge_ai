import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
import hdbscan # type: ignore
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss
import math
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="FoodBridge AI", layout="wide")

st.title("FoodBridge AI Platform")
st.subheader("Intelligent Food Waste Redistribution System")

# Data Loading
DATA_PATH = r"E:\FoodBridge_AI\foodbridge\data\foodmaster_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure coordinates are present (Map city to lat/lon if they weren't in CSV)
    # The uploaded CSV already has latitude and longitude, so we use those.
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Navigation Sidebar
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

# ----------------------------- DASHBOARD & CLUSTERING -----------------------------
if page == "Dashboard Analytics":

    st.header("📊 FoodBridge AI Analytics Dashboard")

    # (Keep your existing Metrics and City/Outlet charts here...)
    total_surplus = df["surplus_kg"].sum()
    total_prepared = df["total_prepared_kg"].sum()
    total_people = df["number_of_people"].sum()
    avg_surplus = df["surplus_kg"].mean()
    potential_meals = int(total_surplus * 3)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Surplus (kg)", f"{total_surplus:,.2f}")
    col2.metric("Total Prepared (kg)", f"{total_prepared:,.2f}")
    col3.metric("People Served", f"{int(total_people):,}")
    col4.metric("Avg Surplus", f"{avg_surplus:.2f}")
    col5.metric("Potential Meals", f"{potential_meals:,}")

    # Visualizations

    c_left, c_right = st.columns(2)

   

    with c_left:

        st.subheader("Surplus by City")

        city_surplus = df.groupby("city")["surplus_kg"].sum().reset_index()

        fig1 = px.bar(city_surplus, x="city", y="surplus_kg", color="city", template="plotly_white")

        st.plotly_chart(fig1, use_container_width=True)



    with c_right:

        st.subheader("Outlet Type Contribution")

        outlet_data = df.groupby("outlet_type")["surplus_kg"].sum().reset_index()

        fig2 = px.pie(outlet_data, values="surplus_kg", names="outlet_type", hole=0.4)

        st.plotly_chart(fig2, use_container_width=True)



    # Map Visualization

    st.subheader("📍 Live Food Surplus Locations")

    fig_map = px.scatter_mapbox(

        df,

        lat="latitude",

        lon="longitude",

        size="surplus_kg",

        color="surplus_kg",

        hover_name="city",

        hover_data=["outlet_type", "surplus_kg"],

        zoom=4,

        mapbox_style="open-street-map"

    )

    st.plotly_chart(fig_map, use_container_width=True)

    # --- ADVANCED CLUSTERING SECTION ---
    st.divider()
    st.header("🔍 Advanced Surplus Clustering")
    
    # 1. Feature Selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Unnamed: 0' in numeric_cols: numeric_cols.remove('Unnamed: 0')
    
    selected_features = st.multiselect(
        "Select Features for Clustering", 
        options=numeric_cols, 
        default=["latitude", "longitude", "surplus_kg", "footfall_count"]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two numeric features.")
    else:
        # Prep Data
        X = df[selected_features].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)

        # 2. Algorithm Selection
        c_alg1, c_alg2 = st.columns([1, 2])
        algo_choice = c_alg1.radio("Algorithm", ["K-Means", "DBSCAN", "HDBSCAN"])
        
        if algo_choice == "K-Means":
            k_val = c_alg2.slider("Clusters (K)", 2, 12, 5)
            model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            df["cluster"] = model.fit_predict(X_scaled)
        elif algo_choice == "DBSCAN":
            eps_val = c_alg2.slider("Epsilon", 0.1, 5.0, 0.5)
            df["cluster"] = DBSCAN(eps=eps_val, min_samples=5).fit_predict(X_scaled)
        else:
            min_c = c_alg2.slider("Min Cluster Size", 2, 100, 10)
            df["cluster"] = hdbscan.HDBSCAN(min_cluster_size=min_c).fit_predict(X_scaled)

        # --- NEW VISUALIZATION SECTION ---
        tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial Map", "📦 Cluster Variance", "📊 Statistical Profiling"])

        with tab1:
            st.subheader("Geospatial Cluster Distribution")
            fig_map = px.scatter_mapbox(
                df, lat="latitude", lon="longitude", color=df["cluster"].astype(str),
                size="surplus_kg", hover_name="city", zoom=4, 
                mapbox_style="open-street-map", height=600
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("3D Cluster Relationship")
                # Use the first 3 selected features for 3D plot
                feat_3d = selected_features[:3] if len(selected_features) >= 3 else selected_features + [selected_features[0]]
                fig_3d = px.scatter_3d(
                    df, x=feat_3d[0], y=feat_3d[1], z=feat_3d[2] if len(feat_3d)>2 else feat_3d[0],
                    color=df["cluster"].astype(str), opacity=0.7,
                    title=f"Relationship: {', '.join(feat_3d)}"
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            with col_b:
                st.subheader("Cluster Size (Location Count)")
                cluster_counts = df["cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Count"]
                fig_bar = px.bar(cluster_counts, x="Cluster", y="Count", color="Cluster", 
                                 text_auto=True, title="Locations per Cluster")
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("Surplus Intensity by Cluster")
            # Box plot shows the spread of surplus in each cluster
            fig_box = px.box(
                df, x=df["cluster"].astype(str), y="surplus_kg", 
                color=df["cluster"].astype(str),
                points="all", title="Surplus Weight Distribution per Cluster"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.write("**Cluster Summary Table**")
            summary = df.groupby("cluster")[selected_features].mean()
            st.dataframe(summary.style.background_gradient(cmap='Blues'))
# ---------------------- FOOD WASTE PREDICTION ----------------------
elif page == "Food Waste Prediction":
    st.header("🔮 Surplus Prediction Engine")
    
    # Note: These paths must point to your local saved model files
    try:
        regressor = joblib.load(r"E:\FoodBridge_AI\foodbridge\saved_models\foodbridge_regressor.pkl")
        classifier = joblib.load(r"E:\FoodBridge_AI\foodbridge\saved_models\waste_classifier.pkl")
        with open(r"E:\FoodBridge_AI\foodbridge\saved_models\model_features.json") as f:
            reg_features = json.load(f)
        with open(r"E:\FoodBridge_AI\foodbridge\saved_models\classifier_features.json") as f:
            clf_features = json.load(f)
    except:
        st.warning("Model files not found. Please ensure .pkl and .json files are in the 'saved_models' directory.")
        st.stop()

    def prepare_input(data, features):
        df_input = pd.DataFrame([data])
        df_input = pd.get_dummies(df_input)
        for col in features:
            if col not in df_input.columns:
                df_input[col] = 0
        return df_input[features]

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    number_of_people = st.sidebar.slider("Number of People", 10, 1000, 100)
    meals_per_person = st.sidebar.slider("Meals per Person", 1.0, 3.0, 1.5)
    footfall_count = st.sidebar.slider("Footfall", 10, 1000, 120)
    total_prepared_kg = st.sidebar.slider("Prepared Food (kg)", 10, 1000, 200)
    rainfall_mm = st.sidebar.slider("Rainfall (mm)", 0, 100, 5)
    temperature_c = st.sidebar.slider("Temperature (°C)", 10, 45, 30)
    
    city = st.sidebar.selectbox("City", df['city'].unique())
    outlet_type = st.sidebar.selectbox("Outlet Type", df['outlet_type'].unique())
    rainfall_cat = st.sidebar.selectbox("Rainfall Category", ["low", "medium", "high"])

    input_data = {
        "number_of_people": number_of_people,
        "meals_per_person": meals_per_person,
        "footfall_count": footfall_count,
        "total_prepared_kg": total_prepared_kg,
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "city": city,
        "outlet_type": outlet_type,
        "rainfall_category": rainfall_cat
    }

    if st.button("Generate Prediction"):
        X_reg = prepare_input(input_data, reg_features)
        X_clf = prepare_input(input_data, clf_features)
        surplus = regressor.predict(X_reg)[0]
        waste_level = classifier.predict(X_clf)[0]
        
        st.success("Analysis Complete")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Surplus (kg)", f"{surplus:.2f}")
        c2.metric("Waste Level Classification", waste_level)

# --------------------- CHATBOT ---------------------
# --- CONFIGURATION ---
# Get your FREE key at: https://aistudio.google.com/
elif page == "FoodBridge AI Assistant":

    st.header("🤖 FoodBridge AI Assistant")
    st.caption("⚡ Powered by Hybrid AI: RAG (BGE) + Smart Intent Detection")
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
    # Use your fallback for local testing if the file is missing
        GEMINI_API_KEY = "YOUR_api_key"

    genai.configure(api_key=GEMINI_API_KEY)

# --- KNOWLEDGE BASE ---
    documents = [
    "Food surplus should be redistributed within six hours of preparation to ensure safety.",
    "All cooked food must be stored below five degrees Celsius to prevent bacterial growth.",
    "NGOs are responsible for the collection of surplus food from verified restaurants.",
    "Volunteers must perform a visual and smell test on food before distribution.",
    "Route optimization should prioritize high-density poverty areas to minimize travel.",
    "Donors are required to sign a food safety disclaimer during the handover process.",
    "Shelters must update their requirement status every 4 hours for fair distribution.",
    "Transportation vehicles should be kept clean and food containers must be airtight."
]

    @st.cache_resource
    def load_resources():
        """Load BGE Embedding model and FAISS Index"""
        # Lightweight model (~100MB) perfect for Hugging Face Free CPU
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        doc_vectors = model.encode(documents, normalize_embeddings=True)
        index = faiss.IndexFlatIP(doc_vectors.shape[1])
        index.add(np.array(doc_vectors).astype('float32'))
        return model, index

    embed_model, faiss_index = load_resources()
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
# --- LOGIC FUNCTIONS ---
    def get_ai_response(user_query):
    # 1. RAG Check (FoodBridge Specifics)
        instruction = "Represent this sentence for searching relevant passages: "
        q_vec = embed_model.encode([instruction + user_query], normalize_embeddings=True).astype('float32')
        dist, indices = faiss_index.search(q_vec, k=1)
    
    # 0.7+ is a strong match for BGE similarity
        if dist[0][0] > 0.7:
            return f"📍 **FoodBridge Protocol:** {documents[indices[0][0]]}", "safety"
    
    # 2. Gemini Fallback (General Chat)
        try:
            response = llm_model.generate_content(
                f"Role: You are the FoodBridge AI Assistant. Be helpful and concise. \nUser: {user_query}"
        )
            return response.text, "general"
        except Exception as e:
            return "I'm currently optimizing routes and my general brain is offline. Ask me about food safety!", "error"

# --- UI LAYOUT ---
    st.title("🤖 FoodBridge AI Assistant")
    st.markdown("---")

# Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User Input
    if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                ans, msg_type = get_ai_response(prompt)
                st.markdown(ans)
    
    # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": ans})

# Sidebar: Helper Tools
    with st.sidebar:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        st.info("This assistant uses a Hybrid RAG + Gemini Flash architecture.")
# ----------------------------- 4. TRANSFORMER-POWERED SENTIMENT ANALYSIS -----------------------------
elif page == "Sentiment Analysis":
    st.header("💬 Advanced Sentiment Analysis")
    st.info("Using a pre-trained DistilBERT model for deep contextual understanding of feedback.")

    from transformers import pipeline

    @st.cache_resource
    def load_sentiment_pipeline():
        # This downloads a model fine-tuned on the SST-2 (Stanford Sentiment Treebank) dataset
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    try:
        sentiment_analyzer = load_sentiment_pipeline()
        
        text = st.text_area("Enter volunteer or donor feedback:", 
                            placeholder="Example: The distribution was delayed, but the food quality was surprisingly good.")

        if st.button("Analyze Sentiment"):
            if text.strip() == "":
                st.warning("Please enter some feedback text.")
            else:
                with st.spinner("Analyzing context..."):
                    # The model returns a list of dicts: [{'label': 'POSITIVE', 'score': 0.99}]
                    result = sentiment_analyzer(text)[0]
                    label = result['label']
                    score = result['score']

                    # UI Styling based on result
                    if label == "POSITIVE":
                        st.balloons()
                        st.success(f"### Result: {label}")
                    else:
                        st.error(f"### Result: {label}")

                    st.metric("Confidence Score", f"{score*100:.2f}%")
                    
                    # Progress bar for visual representation
                    st.write("Model Certainty:")
                    st.progress(score)

                    with st.expander("Why this result?"):
                        st.write("""
                        Unlike basic word-matching, this Transformer model (DistilBERT) 
                        looks at the relationship between every word in your sentence. 
                        It can detect if a positive word is being negated or if the 
                        overall tone is frustrated despite having 'good' words.
                        """)

    except Exception as e:
        st.error("Could not load the Transformer model. Ensure you have 'transformers' and 'torch' installed.")
        st.info("Run: pip install transformers torch")

# ----------------------------- 5. MULTI-LANGUAGE VOLUNTEER TRANSLATOR -----------------------------
elif page == "Language Translation":
    st.header("🌐 Multi-Language Volunteer Translator")
    st.info("Translate instructions and food safety guidelines into local Indian languages.")

    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

    # Dictionary mapping for supported Indian languages in mBART-50
    SUPPORTED_LANGUAGES = {
        "Tamil": "ta_IN",
        "Hindi": "hi_IN",
        "Telugu": "te_IN",
        "Malayalam": "ml_IN",
        "Marathi": "mr_IN",
        "Bengali": "bn_IN"
        
          # Note: verify model support, standard mbart-50-many-to-many supports these
    }

    @st.cache_resource
    def load_translator():
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model

    tokenizer, model = load_translator()

    # Layout for language selection
    col_lang1, col_lang2 = st.columns(2)
    
    with col_lang1:
        src_lang = st.selectbox("Source Language", ["English"], index=0)
        # mBART code for English
        tokenizer.src_lang = "en_XX"

    with col_lang2:
        target_lang_name = st.selectbox("Target Local Language", list(SUPPORTED_LANGUAGES.keys()))
        target_lang_code = SUPPORTED_LANGUAGES[target_lang_name]

    text = st.text_area("Enter text to translate:", 
                        placeholder="e.g., Please ensure the food is distributed to the shelter before 6 PM.")

    if st.button("Translate Now"):
        if text.strip() == "":
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner(f"Translating to {target_lang_name}..."):
                try:
                    # Tokenize input
                    encoded_input = tokenizer(text, return_tensors="pt")
                    
                    # Generate translation using the forced language code
                    generated_tokens = model.generate(
                        **encoded_input, 
                        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
                    )
                    
                    # Decode output
                    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    
                    st.divider()
                    st.subheader(f"Result ({target_lang_name})")
                    st.success(translation)
                    
                    # Helpful feature: Copy-paste helper
                    st.code(translation, language='text')
                    st.caption("You can copy the translated text above for use in WhatsApp or SMS alerts.")
                
                except Exception as e:
                    st.error(f"Translation Error: {e}")

# --------------------- NGO ROUTE OPTIMIZATION ---------------------
elif page == "NGO Route Optimization":
    st.header("🚚 NGO Food Collection Route Optimization")
    st.write("Calculates the most efficient path between cities to minimize travel time.")
    
    locations = df.groupby("city")[["latitude", "longitude"]].mean().reset_index()
    st.dataframe(locations)

    def distance(coord1, coord2):
        return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

    coords = locations[["latitude","longitude"]].values
    n = len(coords)
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = distance(coords[i], coords[j])

    # Simple Nearest Neighbor TSP Algorithm
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
    route.append(0) # Return to start
    
    route_cities = locations.iloc[route]["city"].tolist()
    st.subheader("Optimized Route Sequence")
    st.write(" ➔ ".join(route_cities))
    
    route_df = locations.iloc[route]
    fig = px.line_mapbox(
        route_df,
        lat="latitude",
        lon="longitude",
        hover_name="city",
        zoom=4,
        mapbox_style="open-street-map"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success("This route reduces travel distance and potential food spoilage.")
