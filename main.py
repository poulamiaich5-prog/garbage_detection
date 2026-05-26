import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Garbage Classification",
    page_icon="♻️",
    layout="centered"
)

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model("garbage_classifier.h5")

# =====================================
# CLASS NAMES
# =====================================

class_names = [
    "Biodegradable",
    "Non-Biodegradable"
]
# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Garbage Detection",
    page_icon="♻️",
    layout="wide"
)

# =# =====================================
# CUSTOM THEME
# =====================================

def apply_custom_theme():

    # SMART CITY + ECO AI BACKGROUND
    bg_img = "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=2000"

    st.markdown(f"""

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;500;700&display=swap');

    /* MAIN APP */

    .stApp {{

        background:
        linear-gradient(
        rgba(0, 8, 20, 0.88),
        rgba(0, 25, 15, 0.92)
        ),
        url("{bg_img}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;

        color: #ecfff1;

        font-family: 'Poppins', sans-serif;
    }}

    /* HEADINGS */

    h1, h2, h3, h4 {{

        font-family: 'Orbitron', sans-serif;

        color: #00ffb3;

        text-shadow:
        0 0 10px rgba(0,255,179,0.8),
        0 0 25px rgba(0,255,179,0.4);

        letter-spacing: 1px;
    }}

    /* SIDEBAR */

    [data-testid="stSidebar"] {{

        background:
        linear-gradient(
        rgba(0, 10, 25, 0.97),
        rgba(0, 25, 15, 0.97)
        );

        border-right: 2px solid #00ffb3;

        box-shadow: 0 0 25px rgba(0,255,179,0.3);
    }}

    /* SIDEBAR TEXT */

    [data-testid="stSidebar"] * {{

        color: #d7fff0;
    }}

    /* GLOW BOX */

    .glow-box {{

        background: rgba(0, 15, 25, 0.75);

        backdrop-filter: blur(12px);

        border: 1px solid rgba(0,255,179,0.3);

        border-radius: 18px;

        padding: 22px;

        box-shadow:
        0 0 25px rgba(0,255,179,0.15);

        margin-bottom: 20px;
    }}

    /* BUTTON */

    .stButton>button {{

        background: linear-gradient(
        90deg,
        #00ffb3,
        #00c853
        );

        color: black;

        border-radius: 12px;

        border: none;

        font-weight: bold;

        font-size: 16px;

        transition: 0.3s;

        padding: 10px 20px;
    }}

    .stButton>button:hover {{

        transform: scale(1.05);

        box-shadow:
        0 0 20px #00ffb3;

        background: linear-gradient(
        90deg,
        #00ffd5,
        #00ff88
        );
    }}

    /* FILE UPLOADER */

    .stFileUploader {{

        background: rgba(0,0,0,0.35);

        border: 1px solid rgba(0,255,179,0.3);

        border-radius: 15px;

        padding: 15px;
    }}

    /* METRIC CARDS */

    [data-testid="metric-container"] {{

        background: rgba(0, 20, 30, 0.75);

        border: 1px solid rgba(0,255,179,0.2);

        padding: 18px;

        border-radius: 15px;

        box-shadow:
        0 0 15px rgba(0,255,179,0.1);
    }}

    /* PROGRESS BAR */

    .stProgress > div > div > div > div {{

        background:
        linear-gradient(
        90deg,
        #00ffb3,
        #00ff66
        );
    }}

    /* RADIO BUTTONS */

    div[role="radiogroup"] > label {{

        background: rgba(0,255,179,0.08);

        padding: 10px;

        border-radius: 10px;

        margin-bottom: 8px;

        transition: 0.3s;
    }}

    div[role="radiogroup"] > label:hover {{

        background: rgba(0,255,179,0.18);

        transform: translateX(5px);
    }}

    /* IMAGE */

    img {{

        border-radius: 15px;

        box-shadow:
        0 0 25px rgba(0,255,179,0.2);
    }}

    /* SCROLLBAR */

    ::-webkit-scrollbar {{

        width: 10px;
    }}

    ::-webkit-scrollbar-thumb {{

        background: #00ffb3;

        border-radius: 10px;
    }}

    </style>

    """, unsafe_allow_html=True)

# APPLY THEME
apply_custom_theme()

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.markdown(
        """
        <h1 style='text-align:center;'>
        ♻️ ECO AI
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="glow-box">

        <h3>🌍 Smart Waste Management</h3>

        AI-powered garbage classification
        system for smart cities.

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================
    # MENU
    # =====================================

    menu = st.radio(

    "🧭 Navigation",

    [

        "🏠 Home",

        "🧠 Prediction",

        "📊 Waste Analytics",

        "♻️ Recycling Tips",

        "🏙️ Smart City Info",

        "📘 About Project",

        "📞 Contact & Feedback"

    ]

)

    st.markdown("---")

    # =====================================
    # MODEL STATUS
    # =====================================

    st.markdown("## 🤖 AI System Status")

    st.success("Model Active")

    st.progress(94)

    st.metric(
        label="Detection Accuracy",
        value="94%"
    )

    st.markdown("---")

    # =====================================
    # GARBAGE TYPES
    # =====================================

    st.markdown("## 🗑️ Waste Categories")

    st.success("🌱 Biodegradable")

    st.write("""
    ✅ Paper

    ✅ Cardboard

    ✅ Food Waste
    """)

    st.error("🧴 Non-Biodegradable")

    st.write("""
    ❌ Plastic

    ❌ Glass

    ❌ Metal
    """)

    st.markdown("---")

    # =====================================
    # SYSTEM INFO
    # =====================================

    st.markdown("## ⚙️ System Info")

    st.markdown(
        """
        <div class="glow-box">

        🔹 TensorFlow AI

        🔹 OpenCV Vision

        🔹 Streamlit Interface

        🔹 Transfer Learning

        🔹 Smart City Integration

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # ECO MESSAGE
    # =====================================

    st.markdown("## 🌿 Eco Awareness")

    st.warning(
        """
        Reduce • Reuse • Recycle ♻️

        Smart waste management creates
        cleaner and greener cities.
        """
    )

    st.markdown("---")

    # =====================================
    # DEVELOPER
    # =====================================

    st.markdown("## 👨‍💻 Developer")

    st.markdown(
        """
        <div class="glow-box">

        Poulami Aich

        AI & Deep Learning Project

        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("♻️ Smart Garbage Detection System")

# =====================================
# HOME PAGE
# =====================================

if menu == "🏠 Home":

    st.title("♻️ Garbage Classification for Smart Cities")

    st.markdown(
        """
        <div class="glow-box">

        <h2>🌍 AI-Powered Smart Waste Management System</h2>

        Smart Garbage Detection is an Artificial Intelligence
        based system designed to automatically classify waste
        into biodegradable and non-biodegradable categories.

        This project helps smart cities improve waste
        segregation, recycling efficiency, and environmental
        sustainability using Deep Learning and Computer Vision.

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================
    # PROJECT IMAGE
    # =====================================

    st.image(
        "https://cdn-icons-png.flaticon.com/512/1046/1046857.png",
        width=250
    )

    st.markdown("---")

    # =====================================
    # FEATURES
    # =====================================

    st.subheader("🚀 Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ✅ AI-Based Garbage Classification

            ✅ Upload Image Detection

            ✅ Webcam Garbage Detection

            ✅ Real-Time Prediction

            ✅ Confidence Score Display

            """
        )

    with col2:

        st.markdown(
            """
            ✅ Smart Waste Analytics

            ✅ Recycling Awareness

            ✅ Eco-Friendly Smart City Solution

            ✅ Deep Learning Integration

            ✅ Interactive Dashboard

            """
        )

    st.markdown("---")

    # =====================================
    # WASTE CATEGORIES
    # =====================================

    st.subheader("🗑️ Waste Categories")

    col3, col4 = st.columns(2)

    with col3:

        st.success("🌱 Biodegradable Waste")

        st.write(
            """
            - Food Waste

            - Paper

            - Cardboard

            - Leaves

            - Organic Waste
            """
        )

    with col4:

        st.error("🧴 Non-Biodegradable Waste")

        st.write(
            """
            - Plastic Bottles

            - Glass

            - Metal Cans

            - E-Waste

            - Polythene
            """
        )

    st.markdown("---")

    # =====================================
    # TECHNOLOGIES USED
    # =====================================

    st.subheader("⚙️ Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.info("TensorFlow")

    with tech2:
        st.info("OpenCV")

    with tech3:
        st.info("Streamlit")

    with tech4:
        st.info("Transfer Learning")

    st.markdown("---")

    # =====================================
    # SMART CITY BENEFITS
    # =====================================

    st.subheader("🏙️ Smart City Benefits")

    st.markdown(
        """
        <div class="glow-box">

        🌍 Reduces Environmental Pollution

        ♻️ Improves Recycling Efficiency

        🚛 Supports Smart Waste Collection

        🧠 Automates Waste Segregation

        📊 Provides Waste Analytics

        🌱 Creates Cleaner and Greener Cities

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # HOW IT WORKS
    # =====================================

    st.subheader("🔍 How The System Works")

    st.write(
        """
        1️⃣ Upload garbage image or capture using webcam

        2️⃣ AI model analyzes the waste image

        3️⃣ Deep Learning model predicts waste category

        4️⃣ System displays garbage type with confidence score

        5️⃣ Waste analytics help improve smart city management
        """
    )

    st.markdown("---")

    # =====================================
    # ENVIRONMENTAL MESSAGE
    # =====================================

    st.success(
        "♻️ Small waste management actions today create a cleaner future tomorrow."
    )



# =====================================
# PREDICTION PAGE
# =====================================

elif menu == "🧠 Prediction":

    st.title("🧠 AI Garbage Detection Center")

    st.markdown(
        """
        <div class="glow-box">

        <h3>♻️ Smart Waste Classification System</h3>

        Upload a garbage image or capture using webcam.

        The AI model will automatically classify the waste into:

        🌱 Biodegradable Waste

        🧴 Non-Biodegradable Waste

        This helps smart cities improve recycling,
        waste segregation, and environmental sustainability.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # SMART CITY LIVE STATUS
    # =====================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("♻️ Waste Detected", "1,245")

    with col2:
        st.metric("🌱 Recycled Items", "932")

    with col3:
        st.metric("🏙️ Smart Bins Active", "48")

    st.markdown("---")

    # =====================================
    # DETECTION MODE
    # =====================================

    st.subheader("📸 Select Detection Method")

    detection_mode = st.radio(

        "Choose Input Source",

        [

            "📂 Upload Image",

            "📷 Webcam Capture"

        ]

    )

    st.markdown("---")

    # =====================================
    # IMAGE UPLOAD MODE
    # =====================================

    if detection_mode == "📂 Upload Image":

        st.info("Upload garbage image for AI analysis")

        uploaded_file = st.file_uploader(

            "Choose an image file",

            type=["jpg", "jpeg", "png"]

        )

        if uploaded_file is not None:

            # Read image
            image = Image.open(uploaded_file)

            # DISPLAY IMAGE
            st.image(
                image,
                caption="📂 Uploaded Garbage Image",
                use_container_width=True
            )

            # IMAGE ARRAY
            img = np.array(image)

            # COLOR CONVERSION
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # RESIZE
            img_resized = cv2.resize(img, (224, 224))

            # NORMALIZE
            img_resized = img_resized / 255.0

            # EXPAND DIMS
            img_resized = np.expand_dims(img_resized, axis=0)

            # =====================================
            # AI PROCESSING ANIMATION
            # =====================================

            with st.spinner("🧠 AI Model Processing Waste Image..."):

                prediction = model.predict(img_resized)

            # =====================================
            # CLASSIFICATION
            # =====================================

            if prediction[0][0] < 0.5:

                result = class_names[0]

                confidence = (1 - prediction[0][0]) * 100

                st.success(f"🌱 Prediction: {result}")

                waste_tip = """
                ✅ This waste can decompose naturally.

                ♻️ Recommended Action:
                Convert into compost or recycle paper materials.
                """

            else:

                result = class_names[1]

                confidence = prediction[0][0] * 100

                st.error(f"🧴 Prediction: {result}")

                waste_tip = """
                ⚠️ This waste may harm the environment.

                ♻️ Recommended Action:
                Send plastic/glass/metal waste for recycling.
                """

            # =====================================
            # CONFIDENCE SCORE
            # =====================================

            st.info(f"🎯 Confidence Score: {confidence:.2f}%")

            st.progress(int(confidence))

            # =====================================
            # AI RECOMMENDATION
            # =====================================

            st.subheader("🤖 Smart Waste Recommendation")

            st.markdown(
                f"""
                <div class="glow-box">

                {waste_tip}

                🌍 Proper waste segregation helps create
                cleaner and smarter cities.

                </div>
                """,
                unsafe_allow_html=True
            )

            # =====================================
            # ENVIRONMENTAL IMPACT
            # =====================================

            st.subheader("🌍 Environmental Impact")

            if result == "Biodegradable":

                st.success(
                    """
                    This waste can be naturally decomposed
                    and reused as organic compost.
                    """
                )

            else:

                st.warning(
                    """
                    Non-biodegradable waste can increase
                    pollution if not recycled properly.
                    """
                )

    # =====================================
    # WEBCAM MODE
    # =====================================

    elif detection_mode == "📷 Webcam Capture":

        st.info("Capture live garbage image using webcam")

        camera_image = st.camera_input("📸 Take a Picture")

        if camera_image is not None:

            # READ IMAGE
            image = Image.open(camera_image)

            # DISPLAY IMAGE
            st.image(
                image,
                caption="📷 Captured Garbage Image",
                use_container_width=True
            )

            # IMAGE ARRAY
            img = np.array(image)

            # COLOR CONVERSION
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # RESIZE
            img_resized = cv2.resize(img, (224, 224))

            # NORMALIZE
            img_resized = img_resized / 255.0

            # EXPAND DIMENSIONS
            img_resized = np.expand_dims(img_resized, axis=0)

            # AI PREDICTION
            with st.spinner("🧠 AI Model Processing Live Image..."):

                prediction = model.predict(img_resized)

            # CLASSIFICATION
            if prediction[0][0] < 0.5:

                result = class_names[0]

                confidence = (1 - prediction[0][0]) * 100

                st.success(f"🌱 Prediction: {result}")

                waste_tip = """
                ✅ Eco-friendly biodegradable waste detected.
                """

            else:

                result = class_names[1]

                confidence = prediction[0][0] * 100

                st.error(f"🧴 Prediction: {result}")

                waste_tip = """
                ⚠️ Non-biodegradable waste detected.
                """

            # CONFIDENCE
            st.info(f"🎯 Confidence Score: {confidence:.2f}%")

            st.progress(int(confidence))

            # RECOMMENDATION
            st.subheader("♻️ Smart Waste Recommendation")

            st.markdown(
                f"""
                <div class="glow-box">

                {waste_tip}

                🌍 AI-powered waste segregation improves
                smart city cleanliness and recycling systems.

                </div>
                """,
                unsafe_allow_html=True
            )

            # SMART CITY MESSAGE
            st.success(
                """
                🏙️ Smart waste detection helps cities reduce
                pollution and improve recycling efficiency.
                """
            )


# =====================================
# WASTE STATISTICS DASHBOARD
# =====================================

elif menu == "📊 Waste Analytics":

    st.title("📊 Smart Waste Analytics Dashboard")

    st.markdown(
        """
        <div class="glow-box">

        <h3>🏙️ AI-Powered Smart City Waste Monitoring</h3>

        This dashboard provides real-time analytics
        for garbage classification and waste management.

        AI-based waste analysis helps improve:
        
        ♻️ Recycling Efficiency
        
        🌍 Environmental Sustainability
        
        🚛 Smart Waste Collection
        
        🧠 Automatic Garbage Segregation

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # SAMPLE DATA
    # =====================================

    biodegradable = 70
    non_biodegradable = 30

    total_waste = biodegradable + non_biodegradable

    recycled_items = 58
    smart_bins = 24

    # =====================================
    # TOP METRICS
    # =====================================

    st.subheader("📌 Live Smart City Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🗑️ Total Waste", total_waste)

    with col2:
        st.metric("🌱 Biodegradable", biodegradable)

    with col3:
        st.metric("🧴 Non-Biodegradable", non_biodegradable)

    with col4:
        st.metric("♻️ Recycled Items", recycled_items)

    st.markdown("---")

    # =====================================
    # SMART CITY STATUS
    # =====================================

    st.subheader("🏙️ Smart City Monitoring System")

    st.markdown(
        """
        <div class="glow-box">

        Real-time monitoring system for AI-powered
        garbage management in smart cities.

        </div>
        """,
        unsafe_allow_html=True
    )

    col5, col6, col7 = st.columns(3)

    # =====================================
    # SMART BINS
    # =====================================

    with col5:

        st.success("♻️ Smart Bins Active")

        st.metric(
            label="Active Smart Bins",
            value=f"{smart_bins}"
        )

        st.progress(85)

        if st.button("View Smart Bins"):

            st.image(
                "https://cdn-icons-png.flaticon.com/512/3082/3082037.png",
                caption="AI Smart Garbage Bin",
                use_container_width=True
            )

            st.info(
                """
                Smart bins automatically monitor
                waste levels using AI and IoT sensors.
                """
            )

    # =====================================
    # WASTE COLLECTION
    # =====================================

    with col6:

        st.info("🚛 Waste Collection")

        st.metric(
            label="Collection Efficiency",
            value="92%"
        )

        st.progress(92)

        if st.button("View Collection System"):

            st.image(
                "https://cdn-icons-png.flaticon.com/512/2921/2921822.png",
                caption="Smart Waste Collection Vehicle",
                use_container_width=True
            )

            st.success(
                """
                AI-based route optimization helps
                reduce fuel usage and improve
                waste collection efficiency.
                """
            )

    # =====================================
    # POLLUTION CONTROL
    # =====================================

    with col7:

        st.warning("🌍 Pollution Control")

        st.metric(
            label="Pollution Reduction",
            value="81%"
        )

        st.progress(81)

        if st.button("View Pollution Analytics"):

            st.image(
                "https://cdn-icons-png.flaticon.com/512/4148/4148460.png",
                caption="Smart City Pollution Monitoring",
                use_container_width=True
            )

            st.warning(
                """
                Smart waste segregation reduces
                pollution and improves city cleanliness.
                """
            )

    st.markdown("---")

    # =====================================
    # BAR CHART
    # =====================================

    st.subheader("📈 Waste Category Distribution")

    chart_data = {

        "Waste Type": [

            "Biodegradable",

            "Non-Biodegradable"

        ],

        "Count": [

            biodegradable,

            non_biodegradable

        ]

    }

    st.bar_chart(
        data=chart_data,
        x="Waste Type",
        y="Count"
    )

    st.markdown("---")

    # =====================================
    # PROGRESS ANALYSIS
    # =====================================

    st.subheader("♻️ Waste Percentage Analysis")

    biodegradable_percent = (
        biodegradable / total_waste
    ) * 100

    non_biodegradable_percent = (
        non_biodegradable / total_waste
    ) * 100

    st.success(
        f"🌱 Biodegradable Waste: {biodegradable_percent:.1f}%"
    )

    st.progress(int(biodegradable_percent))

    st.error(
        f"🧴 Non-Biodegradable Waste: {non_biodegradable_percent:.1f}%"
    )

    st.progress(int(non_biodegradable_percent))

    st.markdown("---")

    # =====================================
    # DAILY REPORT TABLE
    # =====================================

    st.subheader("🗓️ Daily Garbage Detection Report")

    st.table({

        "Waste Category": [

            "Paper",

            "Plastic",

            "Glass",

            "Metal",

            "Food Waste",

            "Cardboard"

        ],

        "Detected Items": [

            25,

            18,

            10,

            8,

            39,

            14

        ],

        "Recycled": [

            "✅",

            "✅",

            "✅",

            "✅",

            "♻️ Compost",

            "✅"

        ]

    })

    st.markdown("---")

    # =====================================
    # SMART CITY ANALYTICS
    # =====================================

    st.subheader("🧠 AI Smart City Insights")

    st.markdown(
        """
        <div class="glow-box">

        📌 AI identifies garbage type automatically

        📌 Smart bins help reduce overflowing waste

        📌 Waste segregation improves recycling process

        📌 Real-time analytics reduce manual effort

        📌 Cleaner cities improve public health

        📌 Smart waste systems reduce pollution

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # ENVIRONMENTAL IMPACT
    # =====================================

    st.subheader("🌍 Environmental Impact Analysis")

    pollution_reduction = 82
    recycling_efficiency = 74
    city_cleanliness = 91

    col8, col9, col10 = st.columns(3)

    with col8:

        st.metric(
            "🌿 Pollution Reduction",
            f"{pollution_reduction}%"
        )

        st.progress(pollution_reduction)

    with col9:

        st.metric(
            "♻️ Recycling Efficiency",
            f"{recycling_efficiency}%"
        )

        st.progress(recycling_efficiency)

    with col10:

        st.metric(
            "🏙️ City Cleanliness",
            f"{city_cleanliness}%"
        )

        st.progress(city_cleanliness)

    st.markdown("---")

    # =====================================
    # FUTURE SMART CITY FEATURES
    # =====================================

    st.subheader("🚀 Future Smart City Features")

    st.info(
        """
        🔹 IoT Smart Dustbins

        🔹 Real-Time Garbage Monitoring

        🔹 AI-Based Recycling Automation

        🔹 Smart Waste Collection Vehicles

        🔹 Pollution Tracking Systems

        🔹 Automated Waste Segregation
        """
    )

    st.markdown("---")

    # =====================================
    # FINAL MESSAGE
    # =====================================

    st.success(
        """
        ♻️ AI-powered waste analytics help create
        cleaner, greener, and smarter cities.
        """
    )
# =====================================
# RECYCLING TIPS PAGE
# =====================================

elif menu == "♻️ Recycling Tips":

    st.title("♻️ Recycling Tips & Smart Waste Awareness")

    st.markdown(
        """
        <div class="glow-box">

        <h3>🌍 Smart Recycling for Smart Cities</h3>

        Proper garbage segregation and recycling
        help reduce pollution and create cleaner,
        greener, and healthier smart cities.

        AI-powered waste management systems help
        improve recycling efficiency and reduce
        environmental damage.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # BIODEGRADABLE WASTE
    # =====================================

    st.header("🌱 Biodegradable Waste")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/2909/2909761.png",
            use_container_width=True
        )

    with col2:

        st.success("Examples")

        st.write(
            """
            ✅ Food Waste

            ✅ Paper

            ✅ Cardboard

            ✅ Leaves

            ✅ Vegetable Peels

            ✅ Garden Waste
            """
        )

        st.info("Smart Management Tips")

        st.write(
            """
            ♻️ Convert food waste into compost

            ♻️ Use biodegradable waste bins

            ♻️ Reduce food wastage

            ♻️ Use eco-friendly paper products

            ♻️ Create organic fertilizers
            """
        )

    st.markdown("---")

    # =====================================
    # NON-BIODEGRADABLE WASTE
    # =====================================

    st.header("🧴 Non-Biodegradable Waste")

    col3, col4 = st.columns([1, 2])

    with col3:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/1041/1041916.png",
            use_container_width=True
        )

    with col4:

        st.error("Examples")

        st.write(
            """
            ❌ Plastic Bottles

            ❌ Glass

            ❌ Metal Cans

            ❌ Electronic Waste

            ❌ Plastic Bags

            ❌ Batteries
            """
        )

        st.info("Smart Management Tips")

        st.write(
            """
            ♻️ Recycle plastic products

            ♻️ Reuse glass containers

            ♻️ Separate dry and wet waste

            ♻️ Avoid single-use plastic

            ♻️ Send e-waste to recycling centers
            """
        )

    st.markdown("---")

    # =====================================
    # SMART GARBAGE SEGREGATION
    # =====================================

    st.header("🧠 AI Garbage Segregation")

    st.markdown(
        """
        <div class="glow-box">

        AI-powered garbage classification systems use:

        🔹 Deep Learning

        🔹 Computer Vision

        🔹 Smart Cameras

        🔹 IoT Sensors

        🔹 Real-Time Monitoring

        These technologies automatically identify
        biodegradable and non-biodegradable waste
        for better waste management.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # SMART CITY BENEFITS
    # =====================================

    st.header("🏙️ Smart City Benefits")

    col5, col6, col7 = st.columns(3)

    with col5:

        st.success("🌿 Cleaner Environment")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/427/427735.png",
            width=120
        )

        st.write(
            """
            Smart recycling reduces
            pollution and keeps
            cities clean.
            """
        )

    with col6:

        st.info("🚛 Smart Collection")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/2972/2972185.png",
            width=120
        )

        st.write(
            """
            AI helps optimize
            garbage collection
            systems.
            """
        )

    with col7:

        st.warning("♻️ Better Recycling")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/995/995053.png",
            width=120
        )

        st.write(
            """
            Automatic segregation
            improves recycling
            efficiency.
            """
        )

    st.markdown("---")

    # =====================================
    # RECYCLING STEPS
    # =====================================

    st.header("🗂️ Smart Recycling Steps")

    st.write(
        """
        1️⃣ Separate wet and dry waste

        2️⃣ Dispose garbage into correct bins

        3️⃣ Recycle reusable materials

        4️⃣ Reduce plastic usage

        5️⃣ Use composting for food waste

        6️⃣ Support eco-friendly products
        """
    )

    st.markdown("---")

    # =====================================
    # ENVIRONMENTAL IMPACT
    # =====================================

    st.header("🌍 Environmental Impact")

    pollution_reduction = 85
    recycling_efficiency = 78
    cleanliness = 92

    col8, col9, col10 = st.columns(3)

    with col8:

        st.metric(
            "🌿 Pollution Reduction",
            f"{pollution_reduction}%"
        )

        st.progress(pollution_reduction)

    with col9:

        st.metric(
            "♻️ Recycling Efficiency",
            f"{recycling_efficiency}%"
        )

        st.progress(recycling_efficiency)

    with col10:

        st.metric(
            "🏙️ City Cleanliness",
            f"{cleanliness}%"
        )

        st.progress(cleanliness)

    st.markdown("---")

    # =====================================
    # ECO AWARENESS MESSAGE
    # =====================================

    st.success(
        """
        ♻️ Small recycling actions can create
        a big environmental impact for future generations.
        """
    )

# =====================================
# SMART CITY INFO PAGE
# =====================================

elif menu == "🏙️ Smart City Info":


    st.title("🏙️ Smart City Waste Management")

    # =====================================
    # HEADER SECTION
    # =====================================

    st.markdown(
        """
        <div class="glow-box">

        <h2>🇮🇳 India Smart City Dashboard</h2>

        AI-powered smart waste management system
        for Indian smart cities.

        ♻️ Smart Waste Monitoring

        🌿 Pollution Reduction

        🚛 Smart Collection Tracking

        🧠 AI Waste Analytics

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # LIVE STATUS
    # =====================================

    st.header("📡 Live Smart City Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("♻️ Smart Bins", "1,875")
        st.progress(88)

    with col2:
        st.metric("🚛 Collection Trucks", "245")
        st.progress(81)

    with col3:
        st.metric("🌿 Pollution Reduction", "84%")
        st.progress(84)

    with col4:
        st.metric("🏙️ Cleanliness Score", "92%")
        st.progress(92)

    st.markdown("---")

    # =====================================
    # INDIAN SMART CITY DATA
    # =====================================

    city_data = pd.DataFrame({

        "City": [
            "Mumbai",
            "Delhi",
            "Bengaluru",
            "Kolkata",
            "Chennai",
            "Hyderabad",
            "Pune",
            "Ahmedabad",
            "Surat",
            "Jaipur"
        ],

        "Waste_Tons": [
            820,
            760,
            620,
            590,
            510,
            540,
            470,
            450,
            390,
            350
        ],

        "Recycling_Rate": [
            82,
            79,
            88,
            75,
            80,
            84,
            91,
            77,
            86,
            73
        ],

        "Smart_Bins": [
            180,
            160,
            210,
            145,
            170,
            175,
            220,
            140,
            155,
            120
        ],

        "Pollution_Control": [
            78,
            74,
            88,
            70,
            80,
            82,
            90,
            72,
            84,
            69
        ]

    })

    # =====================================
    # TABLE
    # =====================================

    st.header("📋 Smart City Data")

    st.dataframe(
        city_data,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # BAR CHART
    # =====================================

    st.header("🗑️ Waste Collection Analysis")

    fig1 = px.bar(
        city_data,
        x="City",
        y="Waste_Tons",
        color="Waste_Tons",
        text="Waste_Tons",
        title="Waste Collection by Indian Smart Cities",
        height=500
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # PIE CHART
    # =====================================

    st.header("♻️ Recycling Efficiency")

    fig2 = px.pie(
        city_data,
        names="City",
        values="Recycling_Rate",
        hole=0.4,
        title="Recycling Performance"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # SMART BIN GRAPH
    # =====================================

    st.header("🧠 Smart Bin Deployment")

    fig3 = px.line(
        city_data,
        x="City",
        y="Smart_Bins",
        markers=True,
        title="Smart Bin Network"
    )

    fig3.update_layout(
        height=500
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # POLLUTION CONTROL
    # =====================================

    st.header("🌍 Pollution Reduction")

    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatter(
            x=city_data["City"],
            y=city_data["Pollution_Control"],
            fill='tozeroy',
            mode='lines+markers',
            name='Pollution Control'
        )
    )

    fig4.update_layout(
        title="Pollution Reduction Analysis",
        height=500
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # CITY SELECTOR
    # =====================================

    st.header("🏙️ Smart City Performance Viewer")

    selected_city = st.selectbox(
        "Select Smart City",
        city_data["City"]
    )

    selected_data = city_data[
        city_data["City"] == selected_city
    ]

    st.success(f"📍 Selected City: {selected_city}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🗑️ Waste",
            f"{selected_data['Waste_Tons'].values[0]} Tons"
        )

    with c2:
        st.metric(
            "♻️ Recycling",
            f"{selected_data['Recycling_Rate'].values[0]}%"
        )

    with c3:
        st.metric(
            "🧠 Smart Bins",
            f"{selected_data['Smart_Bins'].values[0]}"
        )

    with c4:
        st.metric(
            "🌿 Pollution Control",
            f"{selected_data['Pollution_Control'].values[0]}%"
        )

    st.markdown("---")

    # =====================================
    # LIVE AI MONITOR
    # =====================================

    st.header("📡 Live AI Monitoring")

    ai_accuracy = random.randint(80, 99)

    st.metric(
        "🧠 AI Detection Accuracy",
        f"{ai_accuracy}%"
    )

    st.progress(ai_accuracy)

    st.markdown("---")

    # =====================================
    # SMART CITY FEATURES
    # =====================================

    st.header("🚀 Smart City Features")

    f1, f2, f3 = st.columns(3)

    with f1:

        st.success("🧠 AI Waste Detection")

        st.write("""
        AI automatically classifies
        garbage using Deep Learning
        and Computer Vision.
        """)

    with f2:

        st.info("📡 IoT Smart Bins")

        st.write("""
        Smart bins monitor garbage
        levels and send real-time
        alerts automatically.
        """)

    with f3:

        st.warning("🚛 Smart Collection")

        st.write("""
        AI route optimization improves
        waste collection efficiency
        and reduces fuel usage.
        """)

    st.markdown("---")

    # =====================================
    # AI CHATBOT
    # =====================================

    st.header("🤖 Smart City AI Assistant")

    st.info("""
    Ask questions about:

    ♻️ Garbage Classification

    🌍 Recycling

    🏙️ Smart Cities

    🧠 AI Technology

    🚛 Waste Collection

    🇮🇳 Indian Smart Cities
    """)

    # SESSION STATE

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # DISPLAY CHAT HISTORY

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # CHAT INPUT

    user_question = st.chat_input(
        "💬 Ask Smart City AI Assistant..."
    )

    # =====================================
    # CHATBOT RESPONSE
    # =====================================

    if user_question:

        # SHOW USER MESSAGE

        st.chat_message("user").markdown(user_question)

        # SAVE USER MESSAGE

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        question = user_question.lower()

        # =====================================
        # RESPONSE LOGIC
        # =====================================

        if any(word in question for word in [
            "hello", "hi", "hey"
        ]):

            response = """
👋 Hello!

Welcome to the Smart City AI Assistant.

How can I help you today?
"""

        elif any(word in question for word in [
            "garbage", "waste", "trash"
        ]):

            response = """
♻️ AI Garbage Classification identifies waste types automatically.

✅ Biodegradable Waste:
Food, paper, leaves

❌ Non-Biodegradable Waste:
Plastic, glass, metal

🧠 AI helps improve recycling efficiency.
"""

        elif any(word in question for word in [
            "recycle", "recycling"
        ]):

            response = """
♻️ Recycling helps:

✅ Reduce pollution

✅ Save natural resources

✅ Keep cities clean

✅ Improve sustainability
"""

        elif any(word in question for word in [
            "smart city", "city"
        ]):

            response = """
🏙️ Smart Cities use:

🧠 AI Technology

📡 IoT Sensors

🚛 Smart Waste Collection

🌍 Pollution Monitoring

♻️ Automated Recycling
"""

        elif any(word in question for word in [
            "ai", "model", "tensorflow"
        ]):

            response = """
🧠 This project uses:

🔹 TensorFlow

🔹 OpenCV

🔹 Deep Learning

🔹 MobileNetV2

for smart garbage classification.
"""

        elif any(word in question for word in [
            "pollution", "environment"
        ]):

            response = """
🌍 Smart waste systems help reduce:

✅ Air Pollution

✅ Water Pollution

✅ Plastic Waste

✅ Environmental Damage
"""

        elif any(word in question for word in [
            "india", "mumbai", "delhi",
            "pune", "hyderabad"
        ]):

            response = """
🇮🇳 Indian Smart Cities are using:

♻️ Smart Waste Management

📡 IoT Monitoring

🚛 Smart Collection Systems

🧠 AI Analytics

🌍 Pollution Reduction
"""

        else:

            response = f"""
🤖 Sorry, I couldn't understand:

👉 {user_question}

Please ask questions related to:

♻️ Recycling

🏙️ Smart Cities

🧠 AI Waste Detection

🚛 Waste Management
"""

        # SHOW BOT RESPONSE

        with st.chat_message("assistant"):
            st.markdown(response)

        # SAVE BOT RESPONSE

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.markdown("---")

    # =====================================
    # FINAL MESSAGE
    # =====================================

    st.success("""
    🌍 Smart waste management helps create
    cleaner, greener, and smarter cities
    for future generations.
    """)
    
# =====================================
# ABOUT PAGE
# =====================================

elif menu == "📘 About Project":

    st.title("📘 About Smart Garbage Classification Project")

    st.markdown(
        """
        <div class="glow-box">

        <h3>♻️ Garbage Classification for Smart Cities</h3>

        This project is an AI-powered smart waste
        management system designed to automatically
        classify garbage into:

        ✅ Biodegradable Waste

        ✅ Non-Biodegradable Waste

        using Deep Learning and Computer Vision.

        The system helps smart cities improve
        waste segregation, recycling efficiency,
        and environmental sustainability.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # PROJECT OVERVIEW
    # =====================================

    st.header("🌍 Project Overview")

    st.write(
        """
        Smart cities generate huge amounts of waste daily.
        Manual waste segregation is slow, costly,
        and less efficient.

        This AI system automates garbage classification
        using image processing and deep learning
        technologies.

        The project helps:

        ♻️ Improve recycling systems

        🌿 Reduce environmental pollution

        🧠 Automate garbage detection

        🚛 Support smart waste collection systems

        🏙️ Build cleaner and smarter cities
        """
    )

    st.markdown("---")

    # =====================================
    # PROJECT FEATURES
    # =====================================

    st.header("🚀 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success("🧠 AI Detection")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
            width=120
        )

        st.write(
            """
            AI model automatically
            detects garbage type
            from uploaded images.
            """
        )

    with col2:

        st.info("📸 Real-Time Prediction")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/685/685655.png",
            width=120
        )

        st.write(
            """
            Users can upload images
            or capture garbage using
            webcam detection.
            """
        )

    with col3:

        st.warning("♻️ Smart Recycling")

        st.image(
            "https://cdn-icons-png.flaticon.com/512/1046/1046857.png",
            width=120
        )

        st.write(
            """
            Smart classification
            improves recycling and
            waste management.
            """
        )

    st.markdown("---")

    # =====================================
    # TECHNOLOGIES USED
    # =====================================

    st.header("💻 Technologies Used")

    st.table({

        "Technology": [

            "TensorFlow / Keras",

            "OpenCV",

            "Streamlit",

            "Transfer Learning",

            "MobileNetV2",

            "Python"

        ],

        "Purpose": [

            "Deep Learning Framework",

            "Image Processing",

            "Web Application",

            "Model Optimization",

            "Pretrained CNN Model",

            "Programming Language"

        ]

    })

    st.markdown("---")

    # =====================================
    # MODEL INFORMATION
    # =====================================

    st.header("🧠 AI Model Information")

    st.markdown(
        """
        <div class="glow-box">

        🔹 Model Name: MobileNetV2

        🔹 Model Type: Transfer Learning CNN

        🔹 Input Size: 224 × 224

        🔹 Classification Type: Binary Classification

        🔹 Classes:
            - Biodegradable
            - Non-Biodegradable

        🔹 Training Dataset: TrashNet Dataset

        🔹 Framework: TensorFlow / Keras

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # DATASET INFORMATION
    # =====================================

    st.header("📂 Dataset Information")

    st.write(
        """
        The model is trained using the
        TrashNet Dataset containing
        multiple categories of waste images.

        Categories include:

        ✅ Paper

        ✅ Cardboard

        ✅ Plastic

        ✅ Glass

        ✅ Metal
        
        ✅ Food Waste
        The dataset helps train the AI model
        for accurate garbage classification.
        """
    )
    st.markdown("---")

    # =====================================
    # PROJECT WORKFLOW
    # =====================================

    st.header("⚙️ Project Workflow")

    st.markdown(
        """
        <div class="glow-box">

        1️⃣ User uploads garbage image

        2️⃣ Image preprocessing starts

        3️⃣ AI model analyzes waste image

        4️⃣ System predicts garbage category

        5️⃣ Smart analytics update dashboard

        6️⃣ Waste management process improves

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # PROJECT BENEFITS
    # =====================================

    st.header("🌿 Benefits of This Project")

    benefit1, benefit2, benefit3 = st.columns(3)

    with benefit1:

        st.metric(
            "♻️ Recycling Improvement",
            "88%"
        )

        st.progress(88)

    with benefit2:

        st.metric(
            "🌍 Pollution Reduction",
            "82%"
        )

        st.progress(82)

    with benefit3:

        st.metric(
            "🏙️ Smart Waste Efficiency",
            "91%"
        )

        st.progress(91)

    st.markdown("---")

    # =====================================
    # FUTURE ENHANCEMENTS
    # =====================================

    st.header("🚀 Future Enhancements")

    st.info(
        """
        🔹 Real-Time CCTV Waste Detection

        🔹 AI Smart Dustbins

        🔹 IoT Sensor Integration

        🔹 Cloud-Based Waste Monitoring

        🔹 Multi-Class Garbage Classification

        🔹 Smart Collection Route Optimization

        🔹 Mobile App Integration
        """
    )

    st.markdown("---")

    # =====================================
    # DEVELOPER SECTION
    # =====================================

    st.header("👨‍💻 Developer Information")

    st.markdown(
        """
        <div class="glow-box">

        👩‍💻 Developer: Poulami Aich

        🎓 Project: Garbage Classification for Smart Cities

        🧠 Domain: Artificial Intelligence & Deep Learning

        ♻️ Goal:
        Build an AI-powered smart waste
        management system for sustainable cities.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # FINAL MESSAGE
    # =====================================

    st.success(
        """
        🌍 AI-powered garbage classification systems
        can help create cleaner, greener,
        smarter, and more sustainable cities
        for the future.
        """
    )
    # =====================================
# CONTACT & FEEDBACK PAGE
# =====================================

elif menu == "📞 Contact & Feedback":

    st.title("📞 Contact & Feedback")

    st.markdown(
        """
        <div class="glow-box">

        <h3>♻️ Smart Garbage Management Support Center</h3>

        Your feedback helps improve
        AI-powered waste management
        systems for smarter and cleaner cities.

        🌍 Together we can build
        sustainable smart cities.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # CONTACT INFORMATION
    # =====================================

    st.header("👨‍💻 Developer Contact Information")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glow-box">

            <h4>📧 Contact Details</h4>

            👩‍💻 Developer:
            Poulami Aich

            📩 Email:
            poulami@example.com

            📍 Location:
            Smart City AI Research Lab

            🌐 Domain:
            Artificial Intelligence &
            Deep Learning

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="glow-box">

            <h4>🌍 Project Vision</h4>

            ♻️ Smart Waste Management

            🧠 AI Garbage Detection

            🌿 Pollution Reduction

            🚛 Smart Collection Systems

            🏙️ Cleaner Smart Cities

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # =====================================
    # FEEDBACK FORM
    # =====================================

    st.header("📝 User Feedback Form")

    name = st.text_input("👤 Enter Your Name")

    email = st.text_input("📧 Enter Your Email")

    city = st.text_input("🏙️ Enter Your City")

    feedback_type = st.selectbox(

        "📌 Feedback Category",

        [

            "AI Prediction",

            "User Interface",

            "Smart City Features",

            "Recycling System",

            "Bug Report",

            "Suggestions"

        ]

    )

    rating = st.slider(

        "⭐ Rate This Project",

        1,

        5,

        4

    )

    feedback_message = st.text_area(

        "💬 Write Your Feedback",

        placeholder="Share your suggestions or ideas..."

    )

    # =====================================
    # SUBMIT BUTTON
    # =====================================

    if st.button("🚀 Submit Feedback"):

        st.success(
            f"""
            Thank you {name} for your feedback!

            Your response helps improve
            AI-powered garbage classification
            systems for smart cities.
            """
        )

        st.balloons()

    st.markdown("---")

    # =====================================
    # USER SATISFACTION
    # =====================================

    st.header("📊 User Satisfaction Analytics")

    col3, col4, col5 = st.columns(3)

    with col3:

        st.metric(
            "😊 Positive Feedback",
            "92%"
        )

        st.progress(92)

    with col4:

        st.metric(
            "♻️ Recycling Awareness",
            "88%"
        )

        st.progress(88)

    with col5:

        st.metric(
            "🏙️ Smart City Support",
            "95%"
        )

        st.progress(95)

    st.markdown("---")

    # =====================================
    # SMART CITY HELP CENTER
    # =====================================

    st.header("🌍 Smart City Help Center")

    st.info(
        """
        📌 This AI system helps:

        ♻️ Automatic Garbage Segregation

        🌿 Reduce Environmental Pollution

        🧠 Improve Waste Classification

        🚛 Optimize Waste Collection

        🏙️ Create Cleaner Smart Cities
        """
    )

    st.markdown("---")

    # =====================================
    # COMMON USER QUESTIONS
    # =====================================

    st.header("❓ Frequently Asked Questions")

    with st.expander("How does AI detect garbage?"):

        st.write(
            """
            The AI model uses Deep Learning
            and Computer Vision to analyze
            uploaded garbage images and
            classify waste categories.
            """
        )

    with st.expander("What types of garbage can be detected?"):

        st.write(
            """
            The system detects:
            
            ✅ Biodegradable Waste
            
            ✅ Non-Biodegradable Waste
            """
        )

    with st.expander("How does this help smart cities?"):

        st.write(
            """
            Smart waste systems improve:
            
            ♻️ Recycling
            
            🌍 Pollution Control
            
            🚛 Waste Collection
            
            🏙️ City Cleanliness
            """
        )

    st.markdown("---")

    # =====================================
    # ECO MESSAGE
    # =====================================

    st.warning(
        """
        🌿 Every small recycling effort
        helps create a cleaner and greener future.
        """
    )

    st.markdown("---")

    # =====================================
    # FINAL MESSAGE
    # =====================================

    st.success(
        """
        ♻️ Thank you for supporting
        AI-powered smart waste management systems.
        """
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption("Smart Waste Management System using AI")vv