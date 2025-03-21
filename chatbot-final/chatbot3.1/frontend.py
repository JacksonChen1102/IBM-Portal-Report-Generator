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

st.set_page_config(page_title="PDF Analysis AI", layout="wide")

st.title("🤖 PDF Analysis AI Assistant")
st.write(
    "📤 Upload a PDF file and select an analysis method to get professional document analysis and visualization reports.")

# Use column layout for better UI
col1, col2 = st.columns([1, 2])

with col1:
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["SWOT", "PESTLE", "MOST", "Sentiment", "WordCloud"]
    )

    # Display analysis type description
    analysis_descriptions = {
        "SWOT": "Strengths, Weaknesses, Opportunities, and Threats analysis",
        "PESTLE": "Political, Economic, Social, Technological, Legal, and Environmental analysis",
        "MOST": "Mission, Objectives, Strategies, and Tactics analysis",
        "Sentiment": "Document sentiment tendency analysis",
        "WordCloud": "Generate document keyword cloud image"
    }

    st.info(analysis_descriptions.get(analysis_type, ""))

    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

    # Initialize session state variables
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""
    if "analysis_text" not in st.session_state:
        st.session_state["analysis_text"] = ""
    if "processing" not in st.session_state:
        st.session_state["processing"] = False
    if "need_rerun" not in st.session_state:
        st.session_state["need_rerun"] = False

with col2:
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📄 Text Content", "📊 Analysis Report", "🔍 Visualization Results"])

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

        # Display extracted text
        if st.session_state["pdf_text"]:
            st.text_area("Extracted Text", st.session_state["pdf_text"], height=400)
        else:
            st.info("Please upload a PDF file to view text content")

# Check if we need to rerun the app
if st.session_state.get("need_rerun", False):
    st.session_state["need_rerun"] = False
    st.rerun()

# Button area
if st.session_state["pdf_text"]:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        # Button: Generate analysis report
        if st.button("Generate Analysis Report", key="generate_report"):
            with st.spinner("Generating report..."):
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
        if st.button("Generate Visualization Chart", key="generate_chart",
                     disabled=(analysis_type == "WordCloud" or not st.session_state.get("analysis_text", ""))):
            with st.spinner("Generating visualization..."):
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
        st.subheader(f"{analysis_type} Analysis Report")
        st.markdown(st.session_state["analysis_text"])

        # If there's PDF data, provide download button
        if st.session_state.get("pdf_data", ""):
            pdf_decoded = base64.b64decode(st.session_state["pdf_data"])
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_decoded,
                file_name=f"{analysis_type}_analysis.pdf",
                mime="application/pdf"
            )
    elif analysis_type == "WordCloud" and "wordcloud" in st.session_state and st.session_state["wordcloud"]:
        st.info("Word cloud is available in the 'Visualization Results' tab")
    else:
        st.info("Please generate an analysis report first")

# Display visualization results in tab3
with tab3:
    if analysis_type == "WordCloud" and "wordcloud" in st.session_state and st.session_state["wordcloud"]:
        wc_data = base64.b64decode(st.session_state["wordcloud"])
        wc_image = Image.open(io.BytesIO(wc_data))
        st.image(wc_image, caption="Generated Word Cloud", use_container_width=True)
        st.download_button(
            label="📥 Download Word Cloud",
            data=wc_data,
            file_name="wordcloud.png",
            mime="image/png"
        )
    elif "chart" in st.session_state and st.session_state["chart"]:
        chart_data = base64.b64decode(st.session_state["chart"])
        chart_image = Image.open(io.BytesIO(chart_data))
        st.image(chart_image, caption=f"{analysis_type} Analysis Visualization", use_container_width=True)
        st.download_button(
            label="📥 Download Visualization Chart",
            data=chart_data,
            file_name=f"{analysis_type}_visualization.png",
            mime="image/png"
        )
    else:
        if analysis_type == "WordCloud":
            st.info("Please generate a word cloud first")
        else:
            st.info("Please generate an analysis report and click 'Generate Visualization Chart'")

# Footer information
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.caption("💡 Analysis of complex documents may take 2-3 minutes, please be patient!")
with footer_col2:
    st.caption("💡 By using this platform, you agree to our terms of service and disclaimer.")