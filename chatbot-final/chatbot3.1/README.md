# PDF AI Interaction App 🤖📄

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.qt.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

An intelligent PDF interaction system with two frontend implementations, powered by a FastAPI backend and Ollama's phi3 model. Ask questions about your PDF documents and receive AI-generated responses!

## 🌟 Features

- **PDF Processing**: Extract and analyze text from PDF documents
- **AI Integration**: Natural language interaction using phi3 model
- **Conversation Memory**: Maintains chat history for context-aware responses
- **Dual Frontend Options**: Choose between desktop (PyQt5) or web (Streamlit) interface
- **Robust Backend**: FastAPI server with error handling and clean output processing

## 🏗️ System Architecture

### Backend (`backend.py`)
```mermaid
graph TD
    A[FastAPI Server] --> B[Generate Endpoint]
    B --> C[Ollama phi3]
    C --> D[ANSI Cleaner]
    D --> E[Response Handler]
```

### Data Flow
```mermaid
graph LR
    A[Upload PDF] --> B[Extract Text]
    B --> C[Ask Question]
    C --> D[Process LLM]
    D --> E[Display Response]
```
```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant L as LLM

    U->>F: Upload PDF
    F->>F: Extract Text
    U->>F: Ask Question
    F->>B: Send Request
    B->>L: Process with phi3
    L->>B: Generate Response
    B->>F: Return Answer
    F->>U: Display Response
```
## 💫 Implementation Details

### Frontend Options

#### 🖥️ PyQt5 Desktop App
![WhatsApp Image 2024-07-16 at 13 27 12_67928848](https://github.com/user-attachments/assets/a754ed38-deda-458a-9b89-6d10e828fd66)
- Native system integration
- Real-time updates
- System file dialogs
- Local resource access
- In-memory state management

#### 🌐 Streamlit Web App
![Screenshot 2025-02-06 at 12 38 35 AM](https://github.com/user-attachments/assets/56b7fd57-e038-4f3e-90f8-50c775cde33e)
- Browser-based interface
- Session state persistence
- Reactive components
- Web file upload
- Automatic UI updates

### Backend Features
- FastAPI with Pydantic models
- Ollama phi3 integration
- ANSI sequence cleaning
- Robust error handling
- Environment configuration

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend:
```bash
uvicorn backend:app --reload
```

4. Launch your preferred frontend:
```bash
# For PyQt5
python pyqt_app.py

# For Streamlit
streamlit run streamlit_app.py
```

## 📊 Technical Comparison

| Feature | PyQt5 | Streamlit |
|---------|-------|-----------|
| Architecture | Desktop Native | Web-Based |
| State Management | In-Memory | Session-Based |
| UI Updates | Real-Time Events | Page Rerenders |
| File Handling | System Dialogs | Web Upload |
| Threading | Synchronous | Asynchronous |

## ⚙️ Key Features
- 10,000 character PDF content limit
- Conversation history tracking
- Error handling and recovery
- Clean architecture design
- Fast response generation
- Multiple frontend options

---
Made with ❤️ using Python, FastAPI, PyQt5/Streamlit, and Ollama
