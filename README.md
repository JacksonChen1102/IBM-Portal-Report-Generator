# 📄 PDF Analysis AI Assistant (IBM-Style)

This is an NLP and visualization-based PDF analysis platform. It uses **Streamlit** for the frontend and **FastAPI** for the backend. The app supports uploading PDF documents and generating analyses including SWOT, PESTLE, MOST, sentiment analysis, and word cloud visualizations.

---

## ✅ Tech Stack

- **Frontend**: Streamlit + IBM-style custom CSS
- **Backend**: FastAPI + LangChain + Ollama
- **Analysis Modules**: SWOT, PESTLE, MOST, Sentiment, WordCloud
- **Vector Store**: ChromaDB
- **Visualization**: matplotlib + wordcloud + reportlab

---

## 📦 Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

Then install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Application

### ✅ Start the Backend API (FastAPI)

Run the following command:

```bash
uvicorn backend:app --reload
```

- Default URL: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`

The backend must be running before using the frontend.

---

### ✅ Start the Frontend App (Streamlit)

Run the Streamlit app:

```bash
streamlit run frontend.py
```

- Default URL: `http://localhost:8501`
- Upload a PDF, select an analysis type, and generate a report or visualization.

---

## 📁 Project Structure

```
├── backend.py         # Backend API and analysis logic
├── frontend.py        # Frontend UI (Streamlit)
├── requirements.txt   # Python dependency file
├── README.md          # Project instructions
```

---

## 📌 Notes

- Always start the backend before launching the frontend.
- If errors occur related to `multipy`, `chromadb`, or `PyMuPDF`, make sure they are listed in your `requirements.txt`.