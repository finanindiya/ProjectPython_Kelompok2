# ========== IMPORT LIBRARIES ============
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# ========== DASHBOARD CONFIG ============
st.set_page_config(
    page_title="Car Sales Dashboard",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded" # Changed to expanded as filters are back in sidebar
)

# ========== TEMA / CSS KUSTOM ==========
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #FFFFFF; /* Light gray background for main app */
        color: #333333;
    }

    .plotly-container, .element-container .stPlotlyChart, .element-container .stAltairChart {
        background-color: white;
        border-radius: 20px;
        padding: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        overflow: hidden;
        border: 1px solid #E0E0E0;
    }
     [data-testid="stPlotlyChart"] > div {
        border-radius: 20px !important;
        overflow: hidden;
    }

    /* Top navigation bar (simulated header for a darker top section) */
    .stApp > header {
        background-color: #0077B6; /* Dark blue for top header */
        height: 70px; /* Adjust height as needed */
        display: flex;
        align-items: center;
        padding: 0 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: white; /* Text color for header */
    }
    .stApp > header h1 { /* Targeting the title in the header if it lands there */
        color: white !important;
        margin: 0;
    }

    .stSidebar {
        background-color: #0077B6 !important; /* Ini adalah properti untuk warna background sidebar */
        border-right: 1px solid #003366;
        color: #FFFFFF; /* Warna teks di sidebar */
    }

    /* Text within sidebar */
    .stSidebar .st-emotion-cache-1we6djp, /* target filter labels */
    .stSidebar .st-emotion-cache-nahz7x, /* target multiselect labels */
    .stSidebar label { /* general label for sidebar inputs */
        color: #FFFFFF !important; /* White text for multiselect labels */
    }

    .stButton>button {
    background-color: #FFFFFF; /* Ini adalah properti untuk warna background tombol */
    color: #001F3F; /* Ini adalah properti untuk warna teks tombol */
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 16px;
    border: none;
    box-shadow: 3px 3px 8px rgba(0,0,0,0.2);
    transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #ADD8E6; /* Ini adalah properti untuk warna background tombol saat di-hover */
        color: #001F3F;
        transform: translateY(-2px);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.3);
    }

    h1, h2, h3, .chart-title {
        color: #1A1A1A; /* Darker heading color for main content */
        font-weight: 600;
    }

    .chart-title {
        font-size: 26px;
        font-weight: 700;
        color: #1A1A1A;
        margin-bottom: 20px;
        text-align: center;
        padding-top: 10px;
    }

    /* Metric Cards - Blue background, white text */
    [data-testid="stMetric"] {
        background-color: #0077B6; /* Blue background for metrics */
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #ddd;
        width: 100%;
        min-width: 200px;
        max-width: 280px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: #FFFFFF; /* Text color for metrics */
    }

    [data-testid="stMetricLabel"] {
        font-size: 30px;  /* Lebih besar dari sebelumnya */
        font-weight: 600;
        color: white;
        margin-bottom: 5px;
        text-align: center;
        }

       [data-testid="stMetricValue"] {
        font-size: 16px;  /* Nilainya juga bisa dibesarkan */
        font-weight: 200;
        color: white;
    }

    .css-1r6dm7m { /* Targets the main content block */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    hr {
        border-top: 2px solid #D1D9E0;
        margin-top: 30px;
        margin-bottom: 30px;
    }

    /* Tab styling - White background for content, blue for active tab */
    .stTabs [data-testid="stTab"] {
        background-color: #E6EEF5;
        border-radius: 10px 10px 0 0;
        margin-right: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        color: #555555;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background-color: #0077B6; /* Active tab blue */
        color: white;
        border-bottom: 3px solid #005080;
        font-weight: 600;
    }

    .stTabs [data-testid="stTabContent"] {
        background-color: white;
        border-radius: 0 0 15px 15px;
        padding: 25px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
        margin-top: -10px;
    }

    /* Custom CSS for a subtle background image */
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/gray-jean.png"); /* More neutral, subtle pattern */
        background-repeat: repeat;
        background-attachment: fixed;
    }

    /* Ensure plot backgrounds are white with subtle shadows */
    .plotly-container {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }

    /* Additional styles for prediction tab */
    .prediction-result {
        background-color: #E6F3FF;
        border-left: 5px solid #0077B6;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .prediction-result h4 {
        color: #001F3F;
        margin-bottom: 10px;
    }
    .prediction-result p {
        font-size: 24px;
        font-weight: bold;
        color: #0077B6;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ========== LOAD DATA ============
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"])
    data["order_month"] = data["Date"].dt.to_period("M").dt.to_timestamp()
    data["order_year"] = data["Date"].dt.year
    bins = [0, 20000, 40000, 60000, 100000, np.inf]
    labels = ['<20K', '20K-40K', '40K-60K', '60K-100K', '>100K']
    data['Price Category'] = pd.cut(data['Price ($)'], bins=bins, labels=labels, right=False, ordered=True)
    return data

try:
    data = load_data(r"C:\Users\finan\OneDrive\Dokumen\Syafina\SMT 6\Pengantar Python\data.csv")
except FileNotFoundError:
    st.error("Error: 'data python.csv' not found. Please ensure the data file is in the correct directory or path is correct.")
    st.stop()

# --- PREDICTION MODEL SETUP ---
# List of features for prediction
prediction_features_list = [
    'Company', 'Engine', 'Transmission', 'Color', 'Body Style',
    'Annual Income', 'Dealer_Region'
]
target = 'Price ($)'

# Drop rows with NaN in relevant columns for prediction model training
prediction_data = data.dropna(subset=prediction_features_list + [target]).copy()

# Initialize LabelEncoders for all categorical features
le_mappers = {}
categorical_features = [
    'Company', 'Engine', 'Transmission', 'Color', 'Body Style', 'Dealer_Region'
]

encoded_features = []
for feature in categorical_features:
    le = LabelEncoder()
    # Fit and transform, store the encoder
    prediction_data[f'{feature}_encoded'] = le.fit_transform(prediction_data[feature])
    le_mappers[feature] = le # Store the fitted encoder for later use
    encoded_features.append(f'{feature}_encoded')

# Combine numeric and encoded categorical features for the model's X
model_features = ['Annual Income'] + encoded_features
X = prediction_data[model_features]
y = prediction_data[target]

# Train a simple Decision Tree Regressor model
@st.cache_resource # Cache the model so it's not retrained on every rerun
def train_model(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Using all available data for training for a demo purpose (in real app, split train/test)
model = train_model(X, y)
# --- END PREDICTION MODEL SETUP ---

# ========== HEADER (Simulated Top Bar) - Remains the same in CSS, but the Python content moves down ===========
st.markdown("<h1 style='text-align: center; color: #1A1A1A;'>🚗 Car Sales Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555555;'>Delving into Sales, Customer Behavior, and Regional Performance</h3>", unsafe_allow_html=True)
st.markdown("---")

# ========== SIDEBAR FILTER ============
st.sidebar.title("🚗 Data Filters")
st.sidebar.markdown("Adjust the parameters below to explore different segments of the data.")

body_style_filter = st.sidebar.multiselect(
    "Select Body Style:",
    options=data["Body Style"].unique(),
    default=data["Body Style"].unique()
)
company_filter = st.sidebar.multiselect(
    "Select Company:",
    options=data["Company"].unique(),
    default=data["Company"].unique()
)
year_filter = st.sidebar.multiselect(
    "Select Year:",
    options=sorted(data["order_year"].unique()),
    default=sorted(data["order_year"].unique())
)

filtered_data = data[
    (data["Body Style"].isin(body_style_filter)) &
    (data["Company"].isin(company_filter)) &
    (data["order_year"].isin(year_filter))
]

if filtered_data.empty:
    st.warning("No data matches the selected filters. Please adjust your selections.")
    st.stop()


# ========== KPI SECTION ============
st.markdown("<h2 style='text-align: center; color: #1A1A1A;'>Key Performance Indicators</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    average_price = filtered_data["Price ($)"].mean()
    st.metric("**Average Car Price**", f"US$ {average_price:,.2f}")

with col2:
    std_price = filtered_data["Price ($)"].std()
    st.metric("**Price Std. Dev.**", f"US$ {std_price:,.2f}")

with col3:
    avg_income = filtered_data["Annual Income"].mean()
    st.metric("**Average Customer Income**", f"US$ {avg_income:,.2f}")

with col4:
    total_sales = len(filtered_data)
    st.metric("**Total Transactions**", f"{total_sales:,} Cars Sold")
st.markdown("---")

# ========== CHARTS SECTION ============
st.markdown("<h2 style='text-align: center; color: #1A1A1A;'>Interactive Insights</h2>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💸Price Distribution", "👤Sales by Gender", "🚘Avg Price by Body Style",
    "⚖Income vs Price", "🌍Sales by Region", "📈Sales Trend", "💡Price Prediction"
])

with tab1:
    st.markdown("<h3 class='chart-title'>Distribution of Car Prices</h3>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(filtered_data, x="Price ($)", nbins=40,
                                title="Histogram of Car Prices",
                                color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_hist.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               font=dict(family="Poppins"), bargap=0.05, title_x=0.25)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_box = px.box(filtered_data, y="Price ($)", title="Boxplot of Car Prices",
                         color_discrete_sequence=["#0077B6"])
        fig_box.update_layout(plot_bgcolor='white', paper_bgcolor='white',title_x=0.25,
                                font=dict(family="Poppins"))
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.markdown("<h3 class='chart-title'>Car Sales by Gender</h3>", unsafe_allow_html=True)
    gender_counts = filtered_data["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    color_map = {
        "Male": "#0077B6",
        "Female": "#EF3E3E"}
    fig_gender = px.pie(gender_counts, names="Gender", values="Count",
                        title="Sales Distribution by Gender", color="Gender", color_discrete_map=color_map)
    fig_gender.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                             title_x=0.35, font=dict(family="Poppins"), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    fig_gender.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_gender, use_container_width=True)

with tab3:
    st.markdown("<h3 class='chart-title'>Average Price by Body Style</h3>", unsafe_allow_html=True)
    avg_price_body = filtered_data.groupby("Body Style")["Price ($)"].mean().reset_index().sort_values(by="Price ($)", ascending=False)
    fig_body = px.bar(avg_price_body, x="Body Style", y="Price ($)",
                      title="Average Car Price by Body Style",
                      color="Price ($)", color_continuous_scale="Blues")
    fig_body.update_layout(plot_bgcolor='white', paper_bgcolor='white',title_x=0.35,
                            font=dict(family="Poppins"),
                            xaxis_title="Body Style", yaxis_title="Average Price ($)")
    st.plotly_chart(fig_body, use_container_width=True)

with tab4:
    st.markdown("<h3 class='chart-title'>Annual Income vs Car Price</h3>", unsafe_allow_html=True)

    income_price_data = filtered_data[(filtered_data["Annual Income"] > 0)]

    plot_type = st.selectbox(
        "Choose Plot Type for Income vs Price:",
        ("Heatmap", "Box Plot (by Price Category)")
    )

    labels = ["Low", "Medium", "High"]
    bins = [0, 20000, 50000, float('inf')]
    income_price_data["Price Category"] = pd.cut(income_price_data["Price ($)"], bins=bins, labels=labels)

    if plot_type == "Heatmap":
        if not income_price_data.empty:
            fig_heatmap = px.density_heatmap(income_price_data, x="Price ($)", y="Annual Income",
                                             nbinsx=50, nbinsy=50,
                                             title="Kepadatan Konsumen Berdasarkan Harga Mobil dan Pendapatan Tahunan (Heatmap)",
                                             color_continuous_scale=["#003f5c", "#0077B6", "#90E0EF"])
            fig_heatmap.update_layout(plot_bgcolor='white', paper_bgcolor='white',title_x=0.1,
                                     font=dict(family="Poppins"),
                                     xaxis_title="Harga Mobil ($)", yaxis_title="Pendapatan Tahunan ($)")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No data available for 'Annual Income vs Car Price' after filtering for positive income.")

    elif plot_type == "Box Plot (by Price Category)":
        if not income_price_data.empty:
            fig_box_income = px.box(income_price_data, x="Price Category", y="Annual Income",
                                     title="Ringkasan Pendapatan Tahunan berdasarkan Kategori Harga Mobil (Box Plot)",)
            fig_box_income.update_layout( plot_bgcolor='white',paper_bgcolor='white',title_x=0.15,font=dict(family="Poppins"),
                                         xaxis_title="Kategori Harga Mobil",yaxis_title="Pendapatan Tahunan ($)",)
            st.plotly_chart(fig_box_income, use_container_width=True)
        else:
            st.info("No data available for 'Annual Income vs Car Price' after filtering for positive income.")

with tab5:
    st.markdown("<h3 class='chart-title'>Sales by Dealer Region</h3>", unsafe_allow_html=True)
    region_sales = filtered_data["Dealer_Region"].value_counts().reset_index()
    region_sales.columns = ["Region", "Sales"]
    fig_region = px.bar(region_sales, x="Region", y="Sales",
                        title="Number of Cars Sold per Region",
                        color="Sales", color_continuous_scale="Blues")
    fig_region.update_layout(plot_bgcolor='white', paper_bgcolor='white', title_x=0.35,
                             font=dict(family="Poppins"),
                             xaxis_title="Dealer Region", yaxis_title="Number of Sales")
    st.plotly_chart(fig_region, use_container_width=True)

with tab6:
    st.markdown("<h3 class='chart-title'>Monthly Sales Trend</h3>", unsafe_allow_html=True)
    trend_data = filtered_data.groupby("order_month")["Price ($)"].sum().reset_index()
    fig_trend = px.line(trend_data, x="order_month", y="Price ($)", markers=True,
                        title="Monthly Sales Trend (Total Revenue)",
                        template="plotly_white")
    fig_trend.update_traces(line_color="#0077B6", marker_color="#0077B6")
    fig_trend.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                            font=dict(family="Poppins"), title_x=0.35,
                            xaxis_title="Month", yaxis_title="Total Sales Revenue ($)")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab7: # NEW TAB FOR PRICE PREDICTION
    st.markdown("<h3 class='chart-title'>🚗 Car Price Prediction</h3>", unsafe_allow_html=True)
    st.write("Enter the car features below to get a predicted price.")

    # Input widgets for prediction
    col_pred1, col_pred2, col_pred3 = st.columns(3) # Use more columns for better layout

    with col_pred1:
        company_pred = st.selectbox(
            "Company",
            options=le_mappers['Company'].classes_ # Use classes from LabelEncoder
        )
        engine_pred = st.selectbox(
            "Engine",
            options=le_mappers['Engine'].classes_
        )
        transmission_pred = st.selectbox(
            "Transmission",
            options=le_mappers['Transmission'].classes_
        )
    with col_pred2:
        color_pred = st.selectbox(
            "Color",
            options=le_mappers['Color'].classes_
        )
        body_style_pred = st.selectbox(
            "Body Style",
            options=le_mappers['Body Style'].classes_
        )
        dealer_region_pred = st.selectbox(
            "Dealer Region",
            options=le_mappers['Dealer_Region'].classes_
        )
    with col_pred3:
        annual_income_pred = st.number_input(
            "Annual Income ($)",
            min_value=float(prediction_data['Annual Income'].min()),
            max_value=float(prediction_data['Annual Income'].max()),
            value=float(prediction_data['Annual Income'].median()),
            step=1000.0
        )

    if st.button("Predict Price"):
        try:
            # Encode selected categorical inputs using the fitted mappers
            company_encoded = le_mappers['Company'].transform([company_pred])[0]
            engine_encoded = le_mappers['Engine'].transform([engine_pred])[0]
            transmission_encoded = le_mappers['Transmission'].transform([transmission_pred])[0]
            color_encoded = le_mappers['Color'].transform([color_pred])[0]
            body_style_encoded = le_mappers['Body Style'].transform([body_style_pred])[0]
            dealer_region_encoded = le_mappers['Dealer_Region'].transform([dealer_region_pred])[0]

            # Prepare input for prediction - Pastikan urutan sesuai dengan model_features
            input_data = pd.DataFrame([[
                annual_income_pred, # Numerik pertama
                company_encoded,
                engine_encoded,
                transmission_encoded,
                color_encoded,
                body_style_encoded,
                dealer_region_encoded
            ]], columns=model_features)

            predicted_price = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="prediction-result">
                <h4>Predicted Car Price:</h4>
                <p>US$ {predicted_price:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please check your inputs and data consistency.")
            st.warning("Ensure all selected options were present in the original training data.")


# ========== DATA TABLE ============
st.markdown("---")
st.subheader("Filtered Data Preview")
st.dataframe(filtered_data, use_container_width=True)

# ========== FOOTER / CREDITS ============
st.markdown("---")
if st.button("Show Credits"):
    st.info("Dashboard developed by *Kelompok 2* | Data Source: *Car Sales Data*")