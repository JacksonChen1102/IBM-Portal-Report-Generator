import streamlit as st
import requests
import base64
from PIL import Image
import io

# Backend API endpoints
UPLOAD_API_URL = "http://127.0.0.1:8000/upload_pdf"
GENERATE_REPORT_URL = "http://127.0.0.1:8000/generate_report"
GENERATE_WORDCLOUD_URL = "http://127.0.0.1:8000/generate_wordcloud"
GENERATE_VISUALIZATION_URL = "http://127.0.0.1:8000/generate_visualization"

# IBM Color Palette
IBM_BLUE = "#0f62fe"  # IBM Blue
IBM_GRAY = "#8d8d8d"  # IBM Gray
IBM_BLACK = "#161616"  # IBM Black
IBM_WHITE = "#ffffff"  # IBM White
IBM_CYAN = "#1192e8"  # IBM Cyan
IBM_GREEN = "#42be65"  # IBM Green
IBM_PURPLE = "#8a3ffc"  # IBM Purple
IBM_MAGENTA = "#ee5396"  # IBM Magenta
IBM_RED = "#da1e28"  # IBM Red
IBM_YELLOW = "#f1c21b"  # IBM Yellow


# Apply IBM aesthetic styling to Streamlit
def apply_ibm_styling():
    st.markdown(f"""
    <style>
        /* IBM Plex Sans font */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

        /* Global font settings */
        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
            color: {IBM_BLACK};
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 600;
            color: {IBM_BLACK};
        }}

        /* Primary buttons - IBM Blue */
        .stButton > button {{
            background-color: {IBM_BLUE};
            color: {IBM_WHITE};
            border: none;
            border-radius: 0;
            font-weight: 500;
            transition: background-color 0.3s;
        }}

        .stButton > button:hover {{
            background-color: #0353e9;
            color: {IBM_WHITE};
        }}

        /* Secondary buttons - outlined style */
        .stDownloadButton > button {{
            background-color: {IBM_WHITE};
            color: {IBM_BLUE};
            border: 1px solid {IBM_BLUE};
            border-radius: 0;
            font-weight: 500;
        }}

        /* Streamlit components styling */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stFileUploader > div,
        .stTextArea > div > div > textarea {{
            border-radius: 0;
            border-color: {IBM_GRAY};
        }}

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 0;
            padding: 8px 16px;
            background-color: {IBM_WHITE};
            border-bottom: 2px solid {IBM_GRAY};
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {IBM_WHITE};
            border-bottom: 2px solid {IBM_BLUE};
        }}

        /* Info box */
        .stAlert {{
            border-radius: 0;
            border-left-color: {IBM_BLUE};
        }}

        /* Table styling */
        .stTable table {{
            border-collapse: collapse;
        }}

        .stTable thead th {{
            background-color: {IBM_BLUE};
            color: {IBM_WHITE};
            font-weight: 500;
        }}

        .stTable tbody tr:nth-child(even) {{
            background-color: #f4f4f4;
        }}

        /* Progress bar */
        .stProgress div {{
            background-color: {IBM_BLUE};
        }}

        /* Footer styling */
        footer {{
            border-top: 1px solid #e0e0e0;
            padding-top: 1rem;
            color: {IBM_GRAY};
            font-size: 0.8rem;
        }}

        /* Adjust padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
    </style>
    """, unsafe_allow_html=True)


# Configure page settings with IBM style
st.set_page_config(
    page_title="PDF Analysis AI | IBM-style",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply IBM styling
apply_ibm_styling()

# Header with IBM-styled logo
st.markdown(f"""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <div style="background-color: {IBM_BLUE}; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center; margin-right: 10px;">
        <span style="color: white; font-weight: bold; font-size: 20px;">AI</span>
    </div>
    <h1 style="margin: 0; font-size: 2rem;">PDF Analysis Assistant</h1>
</div>
<p style="color: {IBM_GRAY}; margin-bottom: 2rem;">Upload a PDF file and select an analysis method to get professional document analysis and visualization reports.</p>
""", unsafe_allow_html=True)

# Main layout with IBM-styled columns
main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    st.markdown(f"<h3 style='color: {IBM_BLACK}; font-weight: 600;'>Document Input</h3>", unsafe_allow_html=True)

    # Analysis type selection with IBM-styled container
    st.markdown(f"""
    <div style="background-color: #f4f4f4; padding: 1rem; margin-bottom: 1rem; border-left: 4px solid {IBM_BLUE};">
        <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: {IBM_BLACK};">Select Analysis Type</h4>
    </div>
    """, unsafe_allow_html=True)

    analysis_type = st.selectbox(
        "",  # Empty label as we're using custom header above
        ["SWOT", "PESTLE", "MOST", "Sentiment", "WordCloud"]
    )

    # Display analysis type description with IBM styling
    analysis_descriptions = {
        "SWOT": "Strengths, Weaknesses, Opportunities, and Threats analysis",
        "PESTLE": "Political, Economic, Social, Technological, Legal, and Environmental analysis",
        "MOST": "Mission, Objectives, Strategies, and Tactics analysis",
        "Sentiment": "Document sentiment tendency analysis",
        "WordCloud": "Generate document keyword cloud image"
    }

    st.info(analysis_descriptions.get(analysis_type, ""))

    # File uploader with IBM styling
    st.markdown(f"""
    <div style="background-color: #f4f4f4; padding: 1rem; margin-top: 1.5rem; margin-bottom: 1rem; border-left: 4px solid {IBM_BLUE};">
        <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: {IBM_BLACK};">Upload PDF Document</h4>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])  # Empty label as we use custom header

    # Initialize session state variables
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""
    if "analysis_text" not in st.session_state:
        st.session_state["analysis_text"] = ""
    if "processing" not in st.session_state:
        st.session_state["processing"] = False
    if "need_rerun" not in st.session_state:
        st.session_state["need_rerun"] = False

with main_col2:
    # Create tabs with IBM styling
    st.markdown(f"<h3 style='color: {IBM_BLACK}; font-weight: 600;'>Analysis Results</h3>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📄 Document Content", "📊 Analysis Report", "🔍 Visualization"])

    with tab1:
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                try:
                    st.session_state["processing"] = True
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(UPLOAD_API_URL, files=files)
                    if response.status_code == 200:
                        st.session_state["pdf_text"] = response.json().get("pdf_text", "")
                        st.success("PDF processed successfully!")
                        st.session_state["processing"] = False
                    else:
                        st.error(f"PDF processing failed: {response.status_code}")
                        st.session_state["processing"] = False
                except Exception as e:
                    st.error(f"Request error: {str(e)}")
                    st.session_state["processing"] = False

        # Display extracted text with IBM styling
        if st.session_state["pdf_text"]:
            st.markdown(f"""
            <div style="background-color: #f4f4f4; padding: 0.5rem; margin-bottom: 0.5rem; border-left: 4px solid {IBM_BLUE};">
                <h4 style="margin: 0; color: {IBM_BLACK};">Extracted Text Content</h4>
            </div>
            """, unsafe_allow_html=True)
            st.text_area("", st.session_state["pdf_text"], height=300, key="text_display")
        else:
            st.markdown(f"""
            <div style="background-color: #f4f4f4; padding: 2rem; text-align: center; color: {IBM_GRAY}; border: 1px dashed {IBM_GRAY};">
                <p>Please upload a PDF document to view content</p>
            </div>
            """, unsafe_allow_html=True)

# Check if we need to rerun the app
if st.session_state.get("need_rerun", False):
    st.session_state["need_rerun"] = False
    st.rerun()

# Action buttons area with IBM styling
if st.session_state["pdf_text"]:
    st.markdown("""<hr style="margin: 2rem 0; border-color: #e0e0e0;">""", unsafe_allow_html=True)


    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        # Button: Generate analysis report
        if st.button("Generate Analysis", key="generate_report"):
            with st.spinner("Generating analysis..."):
                try:
                    if analysis_type == "WordCloud":
                        payload = {"analysis_type": analysis_type, "pdf_text": st.session_state["pdf_text"]}
                        response = requests.post(GENERATE_WORDCLOUD_URL, json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state["wordcloud"] = data.get("wordcloud", "")
                            st.success("Word cloud generated successfully!")
                            st.session_state["need_rerun"] = True
                        else:
                            st.error(f"API error {response.status_code}: {response.text}")
                    else:
                        payload = {"analysis_type": analysis_type, "pdf_text": st.session_state["pdf_text"]}
                        response = requests.post(GENERATE_REPORT_URL, json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state["analysis_text"] = data.get("text", "")
                            st.session_state["pdf_data"] = data.get("pdf", "")
                            st.success("Analysis report generated successfully!")
                            st.session_state["need_rerun"] = True
                        else:
                            st.error(f"API error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Request error: {str(e)}")

    with col4:
        # Button: Generate visualization
        if st.button("Generate Visualization", key="generate_chart",
                     disabled=(analysis_type == "WordCloud" or not st.session_state.get("analysis_text", ""))):
            with st.spinner("Creating visualization..."):
                try:
                    payload = {
                        "analysis_type": analysis_type,
                        "analysis_text": st.session_state["analysis_text"]
                    }
                    response = requests.post(GENERATE_VISUALIZATION_URL, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["chart"] = data.get("chart", "")
                        st.success("Visualization chart generated successfully!")
                        st.session_state["need_rerun"] = True
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Request error: {str(e)}")

    with col5:
        # Button: Reset
        if st.button("Reset All Results", key="reset"):
            for key in ["pdf_text", "analysis_text", "pdf_data", "wordcloud", "chart"]:
                if key in st.session_state:
                    st.session_state[key] = ""
            st.success("All results have been reset")
            st.session_state["need_rerun"] = True

# Display analysis report in tab2
with tab2:
    if st.session_state.get("analysis_text", ""):
        st.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 0.5rem; margin-bottom: 1rem; border-left: 4px solid {IBM_BLUE};">
            <h4 style="margin: 0; color: {IBM_BLACK};">{analysis_type} Analysis Report</h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(st.session_state["analysis_text"])

        # If there's PDF data, provide download button with IBM styling
        if st.session_state.get("pdf_data", ""):
            pdf_decoded = base64.b64decode(st.session_state["pdf_data"])
            st.download_button(
                label="Download PDF Report",
                data=pdf_decoded,
                file_name=f"{analysis_type}_analysis.pdf",
                mime="application/pdf",
                help="Download the report as PDF file"
            )
    elif analysis_type == "WordCloud" and "wordcloud" in st.session_state and st.session_state["wordcloud"]:
        st.info("Word cloud is available in the 'Visualization' tab")
    else:
        st.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 2rem; text-align: center; color: {IBM_GRAY}; border: 1px dashed {IBM_GRAY};">
            <p>Please generate an analysis report first</p>
        </div>
        """, unsafe_allow_html=True)

# Display visualization results in tab3
with tab3:
    if analysis_type == "WordCloud" and "wordcloud" in st.session_state and st.session_state["wordcloud"]:
        st.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 0.5rem; margin-bottom: 1rem; border-left: 4px solid {IBM_BLUE};">
            <h4 style="margin: 0; color: {IBM_BLACK};">Word Cloud Visualization</h4>
        </div>
        """, unsafe_allow_html=True)

        wc_data = base64.b64decode(st.session_state["wordcloud"])
        wc_image = Image.open(io.BytesIO(wc_data))

        st.image(wc_image, caption="", use_container_width=True)

        st.download_button(
            label="Download Word Cloud",
            data=wc_data,
            file_name="wordcloud.png",
            mime="image/png"
        )
    elif "chart" in st.session_state and st.session_state["chart"]:
        st.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 0.5rem; margin-bottom: 1rem; border-left: 4px solid {IBM_BLUE};">
            <h4 style="margin: 0; color: {IBM_BLACK};">{analysis_type} Visualization</h4>
        </div>
        """, unsafe_allow_html=True)

        chart_data = base64.b64decode(st.session_state["chart"])
        chart_image = Image.open(io.BytesIO(chart_data))

        st.image(chart_image, caption="", use_container_width=True)

        st.download_button(
            label="Download Visualization",
            data=chart_data,
            file_name=f"{analysis_type}_visualization.png",
            mime="image/png"
        )
    else:
        if analysis_type == "WordCloud":
            st.markdown(f"""
            <div style="background-color: #f4f4f4; padding: 2rem; text-align: center; color: {IBM_GRAY}; border: 1px dashed {IBM_GRAY};">
                <p>Please generate a word cloud first</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f4f4f4; padding: 2rem; text-align: center; color: {IBM_GRAY}; border: 1px dashed {IBM_GRAY};">
                <p>Please generate an analysis report and click 'Generate Visualization'</p>
            </div>
            """, unsafe_allow_html=True)

# IBM-styled footer
st.markdown("""<hr style="margin: 2rem 0; border-color: #e0e0e0;">""", unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <div style="background-color: {IBM_BLUE}; width: 20px; height: 20px; display: flex; justify-content: center; align-items: center; margin-right: 8px;">
            <span style="color: white; font-weight: bold; font-size: 10px;">AI</span>
        </div>
        <p style="color: {IBM_GRAY}; margin: 0; font-size: 12px;">PDF Analysis AI Assistant © 2025</p>
    </div>
    """, unsafe_allow_html=True)
with footer_col2:
    st.markdown(f"""
    <p style="color: {IBM_GRAY}; margin: 0; font-size: 12px;">By using this PDF Analysis Platform, users acknowledge and agree to the relevant terms and privacy policy.</p>
    """, unsafe_allow_html=True)
with footer_col3:
    st.markdown(f"""
    <p style="color: {IBM_GRAY}; margin: 0; font-size: 12px; text-align: right;">Analysis of complex documents may take 2-3 minutes. Please be patient.</p>
    """, unsafe_allow_html=True)
