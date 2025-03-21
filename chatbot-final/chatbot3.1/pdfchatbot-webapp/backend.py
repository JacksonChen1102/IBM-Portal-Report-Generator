from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import ollama
import io
import base64
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import numpy as np
import textwrap
import re

# langchain & PDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

# wordcloud
from wordcloud import WordCloud, STOPWORDS

# matplotlib for visualization
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

app = FastAPI()

# ---------------------------
# IBM Design System Colors
# ---------------------------
IBM_COLORS = {
    'blue': '#0f62fe',  # IBM Blue
    'gray': '#8d8d8d',  # IBM Gray
    'black': '#161616',  # IBM Black
    'white': '#ffffff',  # IBM White
    'cyan': '#1192e8',  # IBM Cyan
    'green': '#42be65',  # IBM Green
    'purple': '#8a3ffc',  # IBM Purple
    'magenta': '#ee5396',  # IBM Magenta
    'red': '#da1e28',  # IBM Red
    'yellow': '#f1c21b',  # IBM Yellow
    'blue_20': '#d0e2ff',  # IBM Blue - lighter shade
    'blue_40': '#a6c8ff',  # IBM Blue - light shade
    'blue_60': '#78a9ff',  # IBM Blue - medium shade
    'blue_80': '#4589ff',  # IBM Blue - dark shade
    'cyan_20': '#bae6ff',  # IBM Cyan - lighter shade
    'green_20': '#a7f0ba',  # IBM Green - lighter shade
    'purple_20': '#e8daff',  # IBM Purple - lighter shade
    'magenta_20': '#ffafd2',  # IBM Magenta - lighter shade
    'red_20': '#ffd7d9',  # IBM Red - lighter shade
    'yellow_20': '#fcf4d6',  # IBM Yellow - lighter shade
    'gray_10': '#f4f4f4',  # IBM Gray - very light
    'gray_20': '#e0e0e0',  # IBM Gray - lighter
    'gray_30': '#c6c6c6',  # IBM Gray - light
    'gray_60': '#6f6f6f',  # IBM Gray - medium
    'gray_80': '#393939',  # IBM Gray - dark
}


# Set IBM-style matplotlib configuration
def set_ibm_matplotlib_style():
    """Configure matplotlib to use IBM fonts, colors and styling"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = IBM_COLORS['gray_30']
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = IBM_COLORS['gray_20']
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['xtick.color'] = IBM_COLORS['gray_80']
    plt.rcParams['ytick.color'] = IBM_COLORS['gray_80']
    plt.rcParams['text.color'] = IBM_COLORS['black']
    plt.rcParams['axes.labelcolor'] = IBM_COLORS['black']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['savefig.facecolor'] = IBM_COLORS['white']
    plt.rcParams['figure.facecolor'] = IBM_COLORS['white']
    plt.rcParams['figure.figsize'] = (10, 6)

    return plt

# ---------------------------
# 1. Define the prompt template and data model
# ---------------------------

ANALYSIS_TEMPLATES = {
    "SWOT": (
        "Please conduct a detailed SWOT analysis based on the content of the document provided and output it strictly in the following format:\n\n"
        "## Strengths\n"
        "- [Add strength 1 here]\n"
        "- [Add strength 2 here]\n"
        "- [Add more strength points here...]\n\n"

        "## Weaknesses\n"
        "- [Add weakness 1 here]\n"
        "- [Add weakness 2 here]\n"
        "- [Add more weakness points here...]\n\n"

        "## Opportunities\n"
        "- [Add opportunity 1 here]\n"
        "- [Add opportunity 2 here]\n"
        "- [Add more opportunity points here...]\n\n"

        "## Threats\n"
        "- [Add threat point 1 here]\n"
        "- [Add threat point 2 here]\n"
        "- [Add more threat points here...]\n\n"

        "Cautions:\n"
        "1. All points must begin with a dash '-'.\n"
        "2. Provide at least 3 bullet points for each section\n"
        "3. Keep each point to 50 words or less\n"
        "4. Provide specific, data-supported analyses based on document content\n"
        "5. Please do not change the format of the title, it must remain as '## Part Name'!"
    ),

    "PESTLE": (
        "Please perform a detailed PESTLE analysis based on the content of the documentation provided, strictly following the format output below:\n\n"
        "## Political\n"
        "- [Political factors 1]\n"
        "- [Political factors 2]\n"
        "- [Political factors 3]\n\n"

        "## Economic\n"
        "- [Economic factor 1]\n"
        "- [Economic factor 2]\n"
        "- [Economic factor 3]\n\n"

        "## Social\n"
        "- [Social factor 1]\n"
        "- [Social factor 2]\n"
        "- [Social factor 3]\n\n"

        "## Technological\n"
        "- [Technical factor 1]\n"
        "- [Technical factor 2]\n"
        "- [Technical factor 3]\n\n"

        "## Legal\n"
        "- [Legal factor 1]\n"
        "- [Legal factor 2]\n"
        "- [Legal factor 3]\n\n"

        "## Environmental\n"
        "- [Environmental factor 1]\n"
        "- [Environmental factor 2]\n"
        "- [Environmental factor 3]\n\n"

        "Cautions:\n"
        "1. All points must begin with a dash '-'.\n"
        "2. Each section must provide at least 3 bullet points\n"
        "3. Each point should be limited to 50 words or less\n"
        "4. Provide specific, data-supported analyses based on document content\n"
        "5. Please do not change the format of the title, it must remain as '## Part Name'!"
    ),

    "MOST": (
        "Please perform a detailed MOST analysis based on the content of the documentation provided, and strictly follow the format of the output below:\n\n"
        "## Mission\n"
        "- [Mission]\n\n"

        "## Objectives\n"
        "- [Objective 1]\n"
        "- [Objective 2]\n"
        "- [Objective 3]\n\n"

        "## Strategies\n"
        "- [Strategy 1]\n"
        "- [Strategy 2]\n"
        "- [Strategy 3]\n\n"

        "## Tactics\n"
        "- [Tactic 1]\n"
        "- [Tactic 2]\n"
        "- [Tactic 3]\n"
        "- [Tactic4]\n\n"

        "Cautions:\n"
        "1. All points must begin with a dash '-'.\n"
        "2. The mission section should identify the 'core purpose' and 'long-term vision' of the organization. Summarize it in a 'clear and concise' statement\n"
        "3. The objectives and strategies sections each provide at least 3 bullet points\n"
        "4. The tactical section provides at least 4 points\n"
        "5. Keep each point to 50 words or less\n"
        "6. Provide specific, data-supported analyses based on document content\n"
        "7. Please do not change the format of the title, it must remain in the format of '## section name'"
    ),

    "Sentiment": (
        "Please conduct a detailed sentiment analysis based on the content of the document provided, and output it strictly in the following format:\n\n"
        "## Overall sentimental tendencies\n"
        "- Sentiment Score: [insert a value between -1 and 1 here, -1 being extremely negative, 0 being neutral, and 1 being extremely positive]\n"
        "- Overall Rating: [describe overall emotional disposition in a short paragraph here]\n\n"

        "## Positive emotion\n"
        "- [Positive Emotional Description 1]\n"
        "- [Positive Emotional Description 2]\n"
        "- [More positive Emotional Description...]\n\n"

        "## Neutral emotion\n"
        "- [Neutral Emotional Description 1]\n"
        "- [Neutral Emotion Description 2]\n\n"

        "## Negative emotion\n"
        "- [Negative emotion description 1]\n"
        "- [Negative emotion description 2]\n"
        "- [More negative emotion description...]\n\n"

        "## Key Emotional Expressions\n"
        "1. \"[Key emotional sentences taken from the document 1]\"\n"
        "2. \"[Key emotional sentences taken from the document 2]\"\n"
        "3. \"[Key emotional sentences taken from the document 3]\"\n"
        "4. \"[Key emotional sentences taken from the document 4]\"\n"
        "5. \"[Key emotional sentences taken from the document 5]\"\n\n"

        "Caveats:\n"
        "1. Sentiment score must be a value between -1 and 1\n"
        "2. All points must begin with a dash '-'.\n"
        "3. Key emotional expressions must be surrounded by inverted commas and must be taken from the original text\n"
        "4. Provide at least 2 bullet points for each section\n"
        "5. Please do not change the format of the title, it must remain as '## Part Name'!\n"
        "6. Please provide ONLY ONE sentiment score in the 'Sentiment Score' field - do not include multiple score fields"
    )
}


class AnalysisRequest(BaseModel):
    analysis_type: str
    pdf_text: str


class VisualizationRequest(BaseModel):
    analysis_type: str
    analysis_text: str


# ---------------------------
# 2. Auxiliary functions - WordCloud
# ---------------------------

def generate_wordcloud_image(text: str):
    """
    Generate a word cloud with diverse IBM Design Language styling
    """
    # Apply IBM style to matplotlib
    plt = set_ibm_matplotlib_style()

    # Base stopwords from NLTK/WordCloud
    stopwords = set(STOPWORDS)

    # Standard English additions
    standard_english = {
        "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "at", "by", "for",
        "from", "in", "into", "of", "on", "to", "with", "i", "me", "my", "mine", "you", "your",
        "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our",
        "ours", "they", "them", "their", "theirs", "am", "is", "are", "was", "were", "be",
        "been", "being", "do", "does", "did", "have", "has", "had", "can", "could", "will",
        "would", "should", "shall", "may", "might"
    }

    # Months (both full names and abbreviations)
    months = {
        "january", "jan", "february", "feb", "march", "mar", "april", "apr", "may",
        "june", "jun", "july", "jul", "august", "aug", "september", "sept", "sep",
        "october", "oct", "november", "nov", "december", "dec"
    }

    # Time periods
    time_periods = {
        "year", "years", "quarter", "quarters", "month", "months", "annual", "annually",
        "quarterly", "monthly", "fiscal", "fy", "q1", "q2", "q3", "q4", "ytd", "yoy",
        "day", "days", "week", "weeks", "half-year", "half", "semester", "semiannual",
        "biannual", "period", "periods", "date", "dates", "term", "terms"
    }

    # Financial terms
    financial_terms = {
        "revenue", "revenues", "profit", "profits", "loss", "losses", "income", "expense",
        "expenses", "earnings", "cost", "costs", "cash", "asset", "assets", "liability",
        "liabilities", "capital", "equity", "share", "shares", "stock", "stocks", "dividend",
        "dividends", "eps", "ebitda", "gaap", "non-gaap", "nongaap", "operating", "operations",
        "financial", "finance", "financing", "investment", "investments", "investor", "investors",
        "balance", "sheet", "cashflow", "budget", "budgets", "forecast", "forecasts", "outlook",
        "guidance", "projection", "projections", "account", "accounts", "accounting", "audit",
        "audited", "unaudited", "tax", "taxes", "taxation", "fiscal", "credit", "debit", "debt",
        "leverage", "liquidity", "solvency", "amortization", "depreciation", "impairment"
    }

    # Corporate language
    corporate_language = {
        "company", "companies", "corporation", "corporations", "inc", "ltd", "llc", "plc",
        "group", "subsidiary", "subsidiaries", "division", "divisions", "segment", "segments",
        "business", "businesses", "industry", "industries", "market", "markets", "enterprise",
        "enterprises", "organization", "organizations", "firm", "firms", "entity", "entities",
        "corporate", "incorporation", "incorporated", "limited", "holdings", "holding"
    }

    # Reporting language
    reporting_language = {
        "report", "reports", "reporting", "statement", "statements", "disclosure", "disclosures",
        "filing", "filings", "form", "forms", "note", "notes", "section", "sections", "management",
        "board", "director", "directors", "officer", "officers", "executive", "executives",
        "ceo", "cfo", "chairman", "chairwoman", "chairperson", "audit", "auditor", "auditors",
        "committee", "committees", "governance", "compliance", "regulatory", "regulation",
        "regulations", "annual", "quarterly", "interim", "review", "reviews", "footnote", "footnotes"
    }

    # Performance indicators
    performance_indicators = {
        "growth", "increase", "increases", "increased", "decrease", "decreases", "decreased",
        "decline", "declines", "declined", "improvement", "improvements", "improved",
        "performance", "result", "results", "return", "returns", "ratio", "ratios",
        "percentage", "percentages", "basis", "point", "points", "rate", "rates",
        "margin", "margins", "target", "targets", "goal", "goals", "objective", "objectives",
        "kpi", "kpis", "metric", "metrics", "measure", "measures", "indicator", "indicators",
        "efficiency", "effective", "effectiveness", "productivity", "output", "outputs",
        "benchmark", "benchmarks", "comparison", "comparisons", "versus", "vs"
    }

    # Generic business terms
    business_terms = {
        "product", "products", "service", "services", "customer", "customers", "client",
        "clients", "vendor", "vendors", "supplier", "suppliers", "partner", "partners",
        "employee", "employees", "stakeholder", "stakeholders", "strategy", "strategies",
        "strategic", "operation", "operational", "project", "projects", "program", "programs",
        "initiative", "initiatives", "approach", "approaches", "value", "values", "quality",
        "mission", "vision", "core", "commitment", "commitments", "focus", "focused",
        "solution", "solutions", "opportunity", "opportunities", "challenge", "challenges",
        "risk", "risks", "benefit", "benefits", "advantage", "advantages", "capability",
        "capabilities", "competency", "competencies", "infrastructure"
    }

    # Common adjectives in reports
    common_adjectives = {
        "certain", "various", "significant", "significantly", "material", "materially",
        "approximately", "approximate", "estimated", "effective", "effectively", "primarily",
        "substantial", "substantially", "continued", "continuing", "respective", "respectively",
        "strong", "stronger", "strongest", "favorable", "unfavorable", "positive", "negative",
        "lower", "higher", "large", "larger", "small", "smaller", "key", "main", "principal",
        "current", "prior", "previous", "future", "additional", "important", "critical",
        "essential", "relevant", "appropriate", "suitable", "optimal", "optimum", "maximum",
        "minimum", "enhanced", "improved", "increased", "decreased", "reduced", "expanded",
        "leading", "innovative", "competitive", "successful", "unsuccessful", "efficient",
        "inefficient", "consistent", "inconsistent", "sustainable", "unsustainable"
    }

    # Industry jargon (general)
    industry_jargon = {
        "software", "hardware", "platform", "platforms", "solution", "solutions", "technology",
        "technologies", "digital", "data", "user", "users", "banking", "investment", "portfolio",
        "fund", "funds", "loan", "loans", "interest", "mortgage", "mortgages", "patient",
        "patients", "treatment", "treatments", "drug", "drugs", "clinical", "pharmaceutical",
        "therapy", "therapies", "production", "manufacturing", "equipment", "facility",
        "facilities", "plant", "plants", "store", "stores", "retail", "consumer", "consumers",
        "merchandise", "inventory", "supply", "supplies", "chain", "chains", "logistics",
        "procurement", "distribution", "wholesale", "transportation", "shipping"
    }

    # Report metadata
    report_metadata = {
        "page", "pages", "see", "refer", "reference", "references", "note", "notes", "summary",
        "overview", "introduction", "conclusion", "appendix", "appendices", "table", "figure",
        "part", "section", "consolidated", "total", "subtotal", "aggregate", "breakdown",
        "analysis", "chart", "graph", "diagram", "illustration", "exhibit", "presented",
        "reflected", "shown", "demonstrated", "highlighted", "mentioned", "referenced",
        "outlined", "detailed", "elaborated", "discussed", "described", "explained"
    }

    # Forward-looking terms
    forward_looking = {
        "future", "plan", "plans", "planning", "estimate", "estimates", "estimated", "projection",
        "projections", "forecast", "forecasts", "outlook", "expect", "expects", "expected",
        "anticipate", "anticipates", "anticipated", "believe", "believes", "believed", "goal",
        "goals", "target", "targets", "objective", "objectives", "aim", "aims", "intend", "intends",
        "intended", "predict", "predicts", "predicted", "project", "projects", "projected",
        "assume", "assumes", "assumed", "assumption", "assumptions", "prospect", "prospects",
        "prospective", "potential", "possibly", "possible", "likely", "unlikely", "probability"
    }

    # Numbers and units
    numbers_units = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "percent", "percentage", "percentages", "million", "millions", "billion", "billions",
        "trillion", "trillions", "thousand", "thousands", "k", "m", "b", "t", "usd", "eur", "gbp",
        "dollar", "dollars", "euro", "euros", "pound", "pounds", "yen", "yuan", "rmb", "ruble",
        "rupee", "rupees", "number", "numbers", "amount", "amounts", "quantity", "quantities",
        "figure", "figures", "sum", "sums", "total", "totals", "unit", "units", "measurement",
        "measurements", "metric", "metrics", "cent", "cents", "penny", "pennies", "fraction",
        "fractional", "decimal", "decimals", "multiple", "multiples", "double", "triple",
        "quadruple"
    }

    # Add all categories of stopwords
    stopwords.update(standard_english)
    stopwords.update(months)
    stopwords.update(time_periods)
    stopwords.update(financial_terms)
    stopwords.update(corporate_language)
    stopwords.update(reporting_language)
    stopwords.update(performance_indicators)
    stopwords.update(business_terms)
    stopwords.update(common_adjectives)
    stopwords.update(industry_jargon)
    stopwords.update(report_metadata)
    stopwords.update(forward_looking)
    stopwords.update(numbers_units)

    # Convert text to lowercase manually
    text = text.lower()

    # Create a simple rectangular mask without any masking
    # This removes the potential issue with not enough valid pixels
    mask_array = np.ones((500, 900), dtype=np.int32)

    # If text is too short, add some padding to avoid errors
    if len(text.split()) < 20:
        text = text + " " + " ".join(["document"] * 20)

    # IBM color function with diverse color palette
    def ibm_diverse_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        """Color function using a diverse mix of IBM colors"""
        # Define multiple color palettes from IBM color system
        primary_colors = [
            IBM_COLORS['blue'],  # IBM Blue
            IBM_COLORS['purple'],  # IBM Purple
            IBM_COLORS['cyan'],  # IBM Cyan
            IBM_COLORS['green'],  # IBM Green
            IBM_COLORS['magenta']  # IBM Magenta
        ]

        secondary_colors = [
            IBM_COLORS['blue_60'],  # Medium Blue
            IBM_COLORS['purple_20'],  # Light Purple
            IBM_COLORS['cyan_20'],  # Light Cyan
            IBM_COLORS['green_20'],  # Light Green
            IBM_COLORS['magenta_20'],  # Light Magenta
            IBM_COLORS['red_20'],  # Light Red
            IBM_COLORS['yellow_20']  # Light Yellow
        ]

        neutral_colors = [
            IBM_COLORS['gray_80'],  # Dark Gray
            IBM_COLORS['gray_60'],  # Medium Gray
            IBM_COLORS['black']  # IBM Black
        ]

        # Assign colors based on font size (importance)
        if font_size > 45:  # Most important words
            return np.random.choice(primary_colors)
        elif font_size > 35:
            return np.random.choice(primary_colors, p=[0.3, 0.2, 0.2, 0.15, 0.15])
        elif font_size > 25:
            return np.random.choice(primary_colors + secondary_colors,
                                    p=[0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
        else:  # Less important words
            all_colors = primary_colors + secondary_colors + neutral_colors
            probabilities = [0.08, 0.08, 0.08, 0.08, 0.08,  # primary
                             0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,  # secondary
                             0.04, 0.04, 0.04]  # neutral
            return np.random.choice(all_colors, p=probabilities)

    # Create the word cloud with more robust error handling
    try:
        wc = WordCloud(
            width=900,
            height=500,
            background_color=IBM_COLORS['white'],
            color_func=ibm_diverse_color_func,
            stopwords=stopwords,
            collocations=False,
            max_words=100,
            min_word_length=3,
            mask=mask_array,
            prefer_horizontal=0.9,
            margin=10,
            random_state=42
        ).generate(text)
    except ValueError as e:
        # Fallback to simpler configuration if the primary one fails
        print(f"WordCloud generation error: {str(e)}. Trying fallback method...")
        wc = WordCloud(
            width=800,
            height=400,
            background_color=IBM_COLORS['white'],
            stopwords=stopwords,
            max_words=50,
            min_word_length=3,
            random_state=42
        ).generate(text)

    # Create figure with IBM styling
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=IBM_COLORS['white'])
    ax.imshow(wc, interpolation='bilinear')
    ax.set_axis_off()

    # Add IBM-styled title
    plt.suptitle("Document Word Cloud", fontsize=16, color=IBM_COLORS['black'], y=0.98, fontweight='bold')

    # Add subtle branding
    fig.text(0.02, 0.02, "PDF Analysis AI", fontsize=8, color=IBM_COLORS['gray'])

    # Convert to image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="PNG", dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)

    return img_buffer


# ---------------------------
# 3. Auxiliary functions - Visualization Text Utils and SWOT
# ---------------------------

# Improved text wrapping function for IBM-styled visualizations
def add_wrapped_text(ax, text, x, y, width, fontsize=10, max_lines=5):
    """Improved text wrapping function with IBM styling"""
    # Ensure text is not None
    if text is None or text == "":
        text = "(No data)"

    # Ensure parameters are float values
    x, y = float(x), float(y)
    width, fontsize = float(width), float(fontsize)

    # Standardize text beginning, preserving bullet points
    if text.startswith('-') or text.startswith('•'):
        text = text.strip()
    else:
        text = '• ' + text.strip()

    # Adjust wrap width for IBM's cleaner text presentation
    wrap_width = int(70 * width)

    # Use textwrap to wrap text
    wrapped_lines = textwrap.wrap(text, width=wrap_width)

    # Check for empty lines case
    if not wrapped_lines:
        wrapped_lines = ["(No content)"]

    # Limit number of lines and add ellipsis if needed
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines - 1]
        wrapped_lines.append("...")

    # Draw each line of text with IBM styling
    for i, line in enumerate(wrapped_lines):
        ax.text(x, y - i * (fontsize * 0.2), line, fontsize=fontsize,
                horizontalalignment='left', verticalalignment='top',
                wrap=True, color=IBM_COLORS['black'],
                fontfamily='sans-serif')

    # Return next vertical position with appropriate spacing
    return y - len(wrapped_lines) * (fontsize * 0.2) - 0.1


def parse_swot_analysis(text):
    """Parse SWOT analysis text into structured sections"""
    # Handle None values
    if text is None:
        text = ""

    sections = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
    current_section = None

    # Split text into lines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check section titles
        lower_line = line.lower()
        if '##' in line or '#' in line:
            if ('strengths' in lower_line) or ('strength' in lower_line):
                current_section = 'strengths'
                continue
            elif ('weaknesses' in lower_line) or ('weakness' in lower_line):
                current_section = 'weaknesses'
                continue
            elif ('opportunities' in lower_line) or ('opportunit' in lower_line):
                current_section = 'opportunities'
                continue
            elif ('threats' in lower_line) or ('threat' in lower_line):
                current_section = 'threats'
                continue

        # If we're in a section and line starts with bullet, add it
        if current_section and (line.startswith('-') or line.startswith('•')):
            sections[current_section].append(line)
        # If in a section but no bullet, try to identify from content
        elif current_section:
            # Non-bullet line but in current section, add as plain text
            if not ('##' in line or '#' in line):
                sections[current_section].append('• ' + line)
        # Try to identify sections from text without explicit markers
        elif ('strengths' in lower_line) or ('strength' in lower_line):
            sections['strengths'].append('• ' + line)
        elif ('weaknesses' in lower_line) or ('weakness' in lower_line):
            sections['weaknesses'].append('• ' + line)
        elif ('opportunities' in lower_line) or ('opportunit' in lower_line):
            sections['opportunities'].append('• ' + line)
        elif ('threats' in lower_line) or ('threat' in lower_line):
            sections['threats'].append('• ' + line)

    return sections


def generate_swot_visualization(sections):
    """
    Creates a SWOT quadrant diagram with IBM design aesthetic
    """
    # Apply IBM style
    plt = set_ibm_matplotlib_style()

    # Create figure with IBM styling
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=IBM_COLORS['white'])

    # Remove borders for clean IBM look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw cross with IBM styling
    ax.axvline(x=0, color=IBM_COLORS['gray_30'], linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color=IBM_COLORS['gray_30'], linewidth=1.5, alpha=0.8)

    # Configure axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title with IBM styling
    ax.set_title("SWOT Analysis", fontsize=18, color=IBM_COLORS['black'], pad=20, fontweight='bold')

    # Add background colors to quadrants using IBM color palette
    ax.add_patch(plt.Rectangle((-1, 0), 1, 1, fill=True, color=IBM_COLORS['green_20'], alpha=0.7))  # Strengths
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color=IBM_COLORS['red_20'], alpha=0.7))  # Weaknesses
    ax.add_patch(plt.Rectangle((-1, -1), 1, 1, fill=True, color=IBM_COLORS['blue_20'], alpha=0.7))  # Opportunities
    ax.add_patch(plt.Rectangle((0, -1), 1, 1, fill=True, color=IBM_COLORS['purple_20'], alpha=0.7))  # Threats

    # Add quadrant titles with IBM styling
    ax.text(-0.95, 0.95, "Strengths", fontsize=14, fontweight='bold', color=IBM_COLORS['green'])
    ax.text(0.05, 0.95, "Weaknesses", fontsize=14, fontweight='bold', color=IBM_COLORS['red'])
    ax.text(-0.95, -0.05, "Opportunities", fontsize=14, fontweight='bold', color=IBM_COLORS['blue'])
    ax.text(0.05, -0.05, "Threats", fontsize=14, fontweight='bold', color=IBM_COLORS['purple'])

    # Function to add word-wrapped text with IBM styling
    def add_word_wrapped_text(ax, text, x, y, fontsize=10, line_spacing=0.05):
        if not text or not isinstance(text, str):
            return y

        # Clean text and determine bullet style
        if text.startswith('-') or text.startswith('•'):
            clean_text = text[1:].strip()
            bullet = "• "
        else:
            clean_text = text.strip()
            bullet = "• "

        # Add bullet only for first line
        ax.text(x, y, bullet, fontsize=fontsize, ha='left', va='top', color=IBM_COLORS['black'])

        # Process text to fit properly
        words = clean_text.split()
        # Increased chars per line for better text density but still readable
        chars_per_line = 65

        # Build lines
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # Word plus space
            if current_length + word_length <= chars_per_line:
                current_line.append(word)
                current_length += word_length
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length

        # Add final line if not empty
        if current_line:
            lines.append(" ".join(current_line))

        # Draw each line
        text_x = x + 0.03  # Indent text after bullet
        current_y = y

        for line in lines:
            ax.text(text_x, current_y, line, fontsize=fontsize, ha='left', va='top',
                    color=IBM_COLORS['black'])
            current_y -= line_spacing

        # Return final position with extra spacing
        return current_y - 0.03

    # Function to add multiple bullet points with dynamic spacing and IBM styling
    def add_bullet_list(items, x, y, quadrant_height=0.9, fontsize=10):
        current_y = y

        for item in items:
            if item and isinstance(item, str):
                # Estimate lines needed based on text length
                approx_lines = max(1, len(item) // 65)
                spacing_needed = approx_lines * 0.05 + 0.03

                # Check if we have enough space
                if current_y - spacing_needed > (y - quadrant_height):
                    current_y = add_word_wrapped_text(ax, item, x, current_y, fontsize=fontsize)
                else:
                    # Add ellipsis to indicate more content
                    ax.text(x + 0.03, current_y, "...", fontsize=fontsize, ha='left', va='top',
                            color=IBM_COLORS['black'])
                    break

    # Add content to each quadrant with IBM styling
    add_bullet_list(sections["strengths"], -0.95, 0.85, fontsize=10)
    add_bullet_list(sections["weaknesses"], 0.05, 0.85, fontsize=10)
    add_bullet_list(sections["opportunities"], -0.95, -0.15, fontsize=10)
    add_bullet_list(sections["threats"], 0.05, -0.15, fontsize=10)

    # Add subtle IBM branding
    fig.text(0.01, 0.01, "PDF Analysis AI", fontsize=8, color=IBM_COLORS['gray'])

    return fig, ax


# ---------------------------
# 4. Auxiliary functions - PESTLE Analysis
# ---------------------------

def parse_pestle_analysis(text):
    """Parse PESTLE analysis text into structured sections"""
    if text is None:
        text = ""

    sections = {
        "political": [],
        "economic": [],
        "social": [],
        "technological": [],
        "legal": [],
        "environmental": []
    }
    current_section = None

    # Split text into lines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check section headers
        lower_line = line.lower()

        # Identify sections
        if '##' in line or '#' in line:
            if any(x in lower_line for x in ["politic", "政治"]):
                current_section = "political"
                continue
            elif any(x in lower_line for x in ["econom", "经济"]):
                current_section = "economic"
                continue
            elif any(x in lower_line for x in ["social", "社会"]):
                current_section = "social"
                continue
            elif any(x in lower_line for x in ["tech", "技术"]):
                current_section = "technological"
                continue
            elif any(x in lower_line for x in ["legal", "法律"]):
                current_section = "legal"
                continue
            elif any(x in lower_line for x in ["environ", "环境"]):
                current_section = "environmental"
                continue
        else:
            # Try to identify sections from content
            if current_section is None:
                if any(x in lower_line for x in ["politic", "政治"]):
                    current_section = "political"
                    continue
                elif any(x in lower_line for x in ["econom", "经济"]):
                    current_section = "economic"
                    continue
                elif any(x in lower_line for x in ["social", "社会"]):
                    current_section = "social"
                    continue
                elif any(x in lower_line for x in ["tech", "技术"]):
                    current_section = "technological"
                    continue
                elif any(x in lower_line for x in ["legal", "法律"]):
                    current_section = "legal"
                    continue
                elif any(x in lower_line for x in ["environ", "环境"]):
                    current_section = "environmental"
                    continue

        # Collect content for current section
        if current_section and line:
            # If it's a list item, keep original format
            if line.startswith('-') or line.startswith('•'):
                sections[current_section].append(line)
            elif line[0].isdigit() and line[1] in ['.', ')', ' ']:
                # Convert numbered format to dash format
                sections[current_section].append('- ' + line[2:].strip())
            elif not ('##' in line or '#' in line):
                # Non-title text also added
                sections[current_section].append('• ' + line)

    return sections


def generate_pestle_visualization(sections):
    """
    Creates a PESTLE analysis visualization with IBM design language styling
    """
    # Apply IBM style
    plt = set_ibm_matplotlib_style()

    # Set consistent constants for IBM styling
    title_color = IBM_COLORS['black']
    primary_color = IBM_COLORS['blue']
    secondary_color = IBM_COLORS['blue_60']
    content_color = IBM_COLORS['black']

    # PESTLE component definitions
    pestle_data = [
        {"letter": "P", "name": "Political", "color": IBM_COLORS['blue'],
         "insights": sections["political"][:3] if sections["political"] else ["No political factors provided"]},
        {"letter": "E", "name": "Economic", "color": IBM_COLORS['cyan'],
         "insights": sections["economic"][:3] if sections["economic"] else ["No economic factors provided"]},
        {"letter": "S", "name": "Social", "color": IBM_COLORS['green'],
         "insights": sections["social"][:3] if sections["social"] else ["No social factors provided"]},
        {"letter": "T", "name": "Technological", "color": IBM_COLORS['purple'],
         "insights": sections["technological"][:3] if sections["technological"] else [
             "No technological factors provided"]},
        {"letter": "L", "name": "Legal", "color": IBM_COLORS['red'],
         "insights": sections["legal"][:3] if sections["legal"] else ["No legal factors provided"]},
        {"letter": "E", "name": "Environmental", "color": IBM_COLORS['green'],
         "insights": sections["environmental"][:3] if sections["environmental"] else [
             "No environmental factors provided"]}
    ]

    # Reverse order for top-to-bottom display (IBM style guides often prefer this)
    pestle_data.reverse()

    # Calculate row heights with more space
    row_heights = []
    words_per_line = 12  # Number of words per line
    line_spacing = 0.2  # Space between lines
    item_spacing = 0.3  # Space between items

    for item in pestle_data:
        # Calculate height needed for this row with extra spacing
        row_height = 0
        for insight in item["insights"][:3]:  # Maximum 3 insights per category
            if not insight or not isinstance(insight, str):
                continue

            # Count words and calculate lines needed with extra spacing
            words = insight.strip().split()
            lines_needed = (len(words) + words_per_line - 1) // words_per_line  # Ceiling division
            # Add significantly more space per line
            height_needed = lines_needed * line_spacing + item_spacing
            row_height += height_needed

        # Ensure minimum row height with increased padding
        row_height = max(2.0, row_height + 0.5)  # Significantly increased minimum height and padding
        row_heights.append(row_height)

    # Calculate total height
    total_height = sum(row_heights)

    # Create figure with IBM styling
    fig, ax = plt.subplots(figsize=(14, total_height * 1.2), facecolor=IBM_COLORS['white'])

    # Turn off axes for clean IBM look
    ax.axis('off')

    # Function to wrap and display text with IBM styling
    def add_word_limited_text(text, x_start, y_start, fontsize=10):
        if not text or not isinstance(text, str):
            return y_start

        # Clean the text
        if text.startswith('-') or text.startswith('•'):
            clean_text = text[1:].strip()
        else:
            clean_text = text.strip()

        # Add bullet point with IBM styling
        ax.text(x_start, y_start, "•", fontsize=fontsize, ha='left', va='top',
                color=content_color, fontweight='normal')

        # Split into words and group into lines with IBM spacing
        words = clean_text.split()
        text_x = x_start + 0.08  # IBM style often uses slightly larger indents
        current_y = y_start

        for i in range(0, len(words), words_per_line):
            line_words = words[i:i + words_per_line]
            line = " ".join(line_words)
            ax.text(text_x, current_y, line, fontsize=fontsize, ha='left', va='top',
                    color=content_color, fontweight='normal')
            current_y -= line_spacing

        # Return the new y position with extra space for the next item
        return current_y - item_spacing

    # Create PESTLE grid with IBM styling
    current_y = 0  # Start from the top

    for i, (item, row_height) in enumerate(zip(pestle_data, row_heights)):
        # Calculate row positions
        row_top = current_y + row_height
        row_bottom = current_y

        # Letter box (left column) - IBM styling with clean, sharp rectangles
        ax.add_patch(plt.Rectangle((0, row_bottom), 1, row_height, fill=True, color=item["color"],
                                   ec=None, lw=0))
        ax.text(0.5, row_bottom + row_height / 2, item["letter"], ha='center', va='center',
                color=IBM_COLORS['white'], fontsize=36, fontweight='bold')

        # Name box (second column) - IBM styling with slightly lighter shade
        color_with_alpha = item["color"]  # IBM often uses solid colors rather than alpha
        ax.add_patch(plt.Rectangle((1, row_bottom), 2, row_height, fill=True, color=color_with_alpha,
                                   alpha=0.7, ec=None, lw=0))
        ax.text(2, row_bottom + row_height / 2, item["name"], ha='center', va='center',
                color=IBM_COLORS['white'], fontsize=16, fontweight='bold')

        # Content box (right column) - IBM styling with even lighter shade
        ax.add_patch(plt.Rectangle((3, row_bottom), 7, row_height, fill=True, color=color_with_alpha,
                                   alpha=0.2, ec=None, lw=0))

        # Add insights with IBM styling and spacing
        content_y = row_bottom + row_height - 0.3

        for insight in item["insights"][:3]:
            if insight and isinstance(insight, str):
                content_y = add_word_limited_text(insight, 3.2, content_y)

        # Move to the next row
        current_y += row_height

    # Set chart dimensions
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total_height)

    # Add title with IBM styling
    ax.set_title("PESTLE Analysis", fontsize=18, fontweight='bold', color=title_color, pad=20)

    # Add subtle IBM branding
    fig.text(0.01, 0.01, "PDF Analysis AI", fontsize=8, color=IBM_COLORS['gray'])

    return fig, ax


# ---------------------------
# 5. Auxiliary functions - MOST Analysis
# ---------------------------

def parse_most_analysis(text):
    """
    Parse MOST analysis text into structured sections
    """
    # Prevent None values
    if text is None:
        text = ""

    sections = {
        "mission": [],
        "objectives": [],
        "strategies": [],
        "tactics": []
    }
    current_section = None

    # Split text into lines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check section headers
        lower_line = line.lower()

        # Identify each section start
        if '##' in line or '#' in line:
            if any(x in lower_line for x in ["mission", "使命", "任务"]):
                current_section = "mission"
                continue
            elif any(x in lower_line for x in ["objective", "目标"]):
                current_section = "objectives"
                continue
            elif any(x in lower_line for x in ["strateg", "策略"]):
                current_section = "strategies"
                continue
            elif any(x in lower_line for x in ["tactic", "战术", "行动"]):
                current_section = "tactics"
                continue
        else:
            # If line contains keywords but isn't in title format, also identify section
            if current_section is None:
                if any(x in lower_line for x in ["mission", "使命", "任务"]):
                    current_section = "mission"
                    continue
                elif any(x in lower_line for x in ["objective", "目标"]):
                    current_section = "objectives"
                    continue
                elif any(x in lower_line for x in ["strateg", "策略"]):
                    current_section = "strategies"
                    continue
                elif any(x in lower_line for x in ["tactic", "战术", "行动"]):
                    current_section = "tactics"
                    continue

        # Collect current section content
        if current_section and line:
            # If it's a list item, keep original format
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                sections[current_section].append(line)
            elif line[0].isdigit() and line[1] in ['.', ')', ' ']:
                # Convert numbered format to dash format
                sections[current_section].append('- ' + line[2:].strip())
            elif not ('##' in line or '#' in line):
                # Non-title regular text also added
                sections[current_section].append('• ' + line)

    return sections


def generate_most_visualization(sections):
    """
    Creates a MOST analysis visualization with IBM design language styling
    """
    # Apply IBM style
    plt = set_ibm_matplotlib_style()

    # Words per line for better readability with IBM styling
    words_per_line = 10
    line_spacing = 0.25  # Slightly increased for IBM's clean spacing
    item_spacing = 0.5  # More spacing between items

    # Get content for each section
    mission_items = sections["mission"][:2] if sections["mission"] else ["No mission information provided"]
    objectives_items = sections["objectives"][:3] if sections["objectives"] else ["No objective information provided"]
    strategies_items = sections["strategies"][:3] if sections["strategies"] else ["No strategy information provided"]
    tactics_items = sections["tactics"][:4] if sections["tactics"] else ["No tactic information provided"]

    # Set up pyramid dimensions for IBM styling
    pyramid_width = 10
    pyramid_height = 12

    # Create figure with IBM styling
    fig, ax = plt.subplots(figsize=(16, 14), facecolor=IBM_COLORS['white'])
    ax.axis('off')

    # Define coordinates for pyramid
    pyramid_tip_x = pyramid_width / 2
    pyramid_tip_y = pyramid_height
    pyramid_base_left_x = 0
    pyramid_base_right_x = pyramid_width
    pyramid_base_y = 0

    # Calculate divider heights for a proper pyramid
    divider_heights = [
        pyramid_height * 0.25,  # 25% up - divides Tactics and Strategies
        pyramid_height * 0.50,  # 50% up - divides Strategies and Objectives
        pyramid_height * 0.75,  # 75% up - divides Objectives and Mission
    ]

    # Calculate points for each divider line
    divider_points = []
    for height in divider_heights:
        # Calculate width at this height (narrower as we go up)
        width_at_height = pyramid_width * (1 - height / pyramid_height)
        left_x = (pyramid_width - width_at_height) / 2
        right_x = left_x + width_at_height
        divider_points.append([(left_x, height), (right_x, height)])

    # Define IBM colors for each section
    colors = [
        IBM_COLORS['blue'],  # Mission (top) - IBM blue
        IBM_COLORS['purple'],  # Objectives - IBM purple
        IBM_COLORS['cyan'],  # Strategies - IBM cyan
        IBM_COLORS['green']  # Tactics (bottom) - IBM green
    ]

    # Create polygon patches for each section with IBM styling
    # Mission section (top tip)
    mission_polygon = plt.Polygon([
        (pyramid_tip_x, pyramid_tip_y),  # Tip
        divider_points[2][0], divider_points[2][1]  # Third divider
    ], color=colors[0], alpha=0.8)
    ax.add_patch(mission_polygon)

    # Objectives section
    objectives_polygon = plt.Polygon([
        divider_points[2][0], divider_points[2][1],  # Third divider
        divider_points[2][1],  # Third divider right
        divider_points[1][1],  # Second divider right
        divider_points[1][0]  # Second divider left
    ], color=colors[1], alpha=0.8)
    ax.add_patch(objectives_polygon)

    # Strategies section
    strategies_polygon = plt.Polygon([
        divider_points[1][0], divider_points[1][1],  # Second divider
        divider_points[1][1],  # Second divider right
        divider_points[0][1],  # First divider right
        divider_points[0][0]  # First divider left
    ], color=colors[2], alpha=0.8)
    ax.add_patch(strategies_polygon)

    # Tactics section (bottom base)
    tactics_polygon = plt.Polygon([
        divider_points[0][0], divider_points[0][1],  # First divider
        divider_points[0][1],  # First divider right
        (pyramid_base_right_x, pyramid_base_y),  # Bottom right
        (pyramid_base_left_x, pyramid_base_y)  # Bottom left
    ], color=colors[3], alpha=0.8)
    ax.add_patch(tactics_polygon)

    # Add section numbers with IBM styling (white circular badges)
    # IBM often uses clean numbered badges for sequential processes
    for i, y_pos in enumerate([
        (divider_heights[2] + pyramid_height) / 2,  # Mission position
        (divider_heights[1] + divider_heights[2]) / 2,  # Objectives position
        (divider_heights[0] + divider_heights[1]) / 2,  # Strategies position
        divider_heights[0] / 2  # Tactics position
    ]):
        # Draw white circular badge with IBM styling
        circle = plt.Circle((pyramid_width / 2, y_pos), 0.4,
                            color='white', ec=colors[i], lw=2, zorder=10)
        ax.add_patch(circle)
        # Add number with IBM styling
        ax.text(pyramid_width / 2, y_pos, f"{i + 1:02d}", ha='center', va='center',
                color=colors[i], fontsize=16, fontweight='bold', zorder=11)

    # Function to process and display a section with IBM styled bullet points
    def add_section_with_items(title, items, y_position, content_x, fontsize=11):
        # Add section title with IBM styling
        ax.text(content_x, y_position, title, fontsize=16, fontweight='bold',
                color=IBM_COLORS['black'])

        # Start content below title with good spacing
        content_y = y_position - 0.8

        # Add each bullet point with IBM styling
        for item in items:
            if not item or not isinstance(item, str):
                continue

            # Clean text
            if item.startswith('-') or item.startswith('•'):
                clean_text = item[1:].strip()
            else:
                clean_text = item.strip()

            # Add bullet point with IBM styling
            ax.text(content_x, content_y, "•", fontsize=fontsize, ha='left', va='top',
                    color=IBM_COLORS['black'])

            # Split into words and group into lines with IBM spacing
            words = clean_text.split()
            text_x = content_x + 0.4  # Indent after bullet - IBM prefers clean indentation
            current_y = content_y

            # Process each line of this bullet point with IBM spacing
            for i in range(0, len(words), words_per_line):
                line_words = words[i:i + words_per_line]
                line = " ".join(line_words)
                ax.text(text_x, current_y, line, fontsize=fontsize, ha='left', va='top',
                        color=IBM_COLORS['black'])
                current_y -= line_spacing

            # Move to next bullet point with increased spacing
            content_y = current_y - item_spacing

    # Content positioning in IBM style
    content_x = pyramid_width + 1.5

    # Add sections with IBM styling and positioning
    add_section_with_items("Mission", mission_items, pyramid_height + 0.5, content_x)
    add_section_with_items("Objectives", objectives_items, pyramid_height - 3.5, content_x)
    add_section_with_items("Strategies", strategies_items, pyramid_height - 7.5, content_x)
    add_section_with_items("Tactics", tactics_items, pyramid_height - 11.5, content_x)

    # Set chart dimensions
    ax.set_xlim(-1, pyramid_width + 15)
    ax.set_ylim(-2, pyramid_height + 3)

    # Set title with IBM styling
    ax.set_title("MOST Analysis", fontsize=18, fontweight='bold', color=IBM_COLORS['black'], pad=20)

    # Add subtle IBM branding
    fig.text(0.01, 0.01, "PDF Analysis AI", fontsize=8, color=IBM_COLORS['gray'])

    return fig, ax


# ---------------------------
# 6. Auxiliary functions - Sentiment Analysis
# ---------------------------

def parse_sentiment_analysis(text):
    """
    Extract sentiment score value from sentiment analysis text with IBM-styled formatting
    """
    # Prevent None value
    if text is None:
        text = ""

    # Default sentiment score
    score = 0
    found_scores = []

    # Find sentiment score line(s)
    lines = text.split('\n')

    for line in lines:
        line = line.strip().lower()
        # Find lines containing score indicators
        if ("sentiment score" in line or "score" in line) and ":" in line:
            # Extract value after colon
            try:
                value_part = line.split(":", 1)[1].strip()
                # Use regex to extract -1 to 1 value
                score_matches = re.findall(r'-?\d+\.?\d*', value_part)
                if score_matches:
                    score_value = float(score_matches[0])
                    # Ensure score is between -1 and 1
                    score_value = max(-1, min(1, score_value))
                    found_scores.append(score_value)
            except (ValueError, IndexError):
                continue

    # If multiple scores were found, use the non-zero one if available, otherwise use the last one
    if found_scores:
        non_zero_scores = [s for s in found_scores if s != 0]
        if non_zero_scores:
            score = non_zero_scores[0]  # Use the first non-zero score
        else:
            score = found_scores[-1]  # Use the last score if all are zero

    # If no explicit score found, try to estimate from text
    if score == 0 and not found_scores:
        # Simple count of positive and negative keywords
        positive_keywords = ["positive", "positively", "good", "excellent", "favorable", "optimistic", "积极", "正面",
                             "良好"]
        negative_keywords = ["negative", "negatively", "bad", "poor", "unfavorable", "pessimistic", "消极", "负面",
                             "糟糕"]

        text_lower = text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_keywords)
        negative_count = sum(text_lower.count(word) for word in negative_keywords)

        if positive_count > 0 or negative_count > 0:
            total = positive_count + negative_count
            score = (positive_count - negative_count) / total

    return score


def generate_sentiment_visualization(score):
    """
    Creates an improved sentiment analysis visualization with IBM design language styling
    """
    # Apply IBM style
    plt = set_ibm_matplotlib_style()

    # Create a clean IBM-styled figure
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=IBM_COLORS['white'])

    # Turn off axis for clean IBM look
    ax.axis('off')

    # Ensure score is within bounds
    clamped_score = max(-1, min(1, score))

    # Define IBM colors for sentiment regions
    neg_color = IBM_COLORS['red_20']  # Light red for negative region
    neu_color = IBM_COLORS['gray_10']  # Light gray for neutral region
    pos_color = IBM_COLORS['green_20']  # Light green for positive region

    # Define IBM accent colors for the indicator and text
    indicator_color = IBM_COLORS['blue']  # IBM blue for the indicator
    neg_text_color = IBM_COLORS['red']  # IBM red for negative text
    neu_text_color = IBM_COLORS['gray_80']  # IBM dark gray for neutral text
    pos_text_color = IBM_COLORS['green']  # IBM green for positive text

    # Draw the gauge background - more refined and IBM-styled
    gauge_y = 0
    gauge_height = 0.3

    # Create gradient regions with clean IBM styling
    # Negative region (-1 to -0.1)
    ax.add_patch(plt.Rectangle((-1, gauge_y - gauge_height / 2), 0.9, gauge_height,
                               color=neg_color, alpha=0.8, ec=IBM_COLORS['gray_20'], lw=0.5))

    # Neutral region (-0.1 to 0.1)
    ax.add_patch(plt.Rectangle((-0.1, gauge_y - gauge_height / 2), 0.2, gauge_height,
                               color=neu_color, alpha=0.8, ec=IBM_COLORS['gray_20'], lw=0.5))

    # Positive region (0.1 to 1)
    ax.add_patch(plt.Rectangle((0.1, gauge_y - gauge_height / 2), 0.9, gauge_height,
                               color=pos_color, alpha=0.8, ec=IBM_COLORS['gray_20'], lw=0.5))

    # Add elegant tick marks with IBM styling
    tick_height = 0.05
    for x_pos in [-1, -0.5, 0, 0.5, 1]:
        # Draw tick mark
        ax.plot([x_pos, x_pos], [gauge_y - gauge_height / 2 - tick_height, gauge_y - gauge_height / 2],
                color=IBM_COLORS['gray_60'], lw=1)
        # Add value label
        ax.text(x_pos, gauge_y - gauge_height / 2 - tick_height - 0.1, f"{x_pos:.1f}",
                ha='center', fontsize=10, color=IBM_COLORS['gray_60'])

    # Create a more elegant indicator (inspired by IBM's design language)
    # Instead of a basic circle, use a combination of shapes for more visual interest

    # Position the indicator at the score position
    indicator_x = clamped_score
    indicator_y = gauge_y

    # Draw a vertical line from the gauge to the indicator
    ax.plot([indicator_x, indicator_x],
            [gauge_y - gauge_height / 2, gauge_y + gauge_height / 2 + 0.1],
            color=indicator_color, lw=2, zorder=10)

    # Draw an IBM-styled indicator (diamond with circle)
    diamond_size = 0.08
    diamond = plt.Polygon([
        (indicator_x, indicator_y + gauge_height / 2 + 0.1 + diamond_size),
        (indicator_x + diamond_size, indicator_y + gauge_height / 2 + 0.1),
        (indicator_x, indicator_y + gauge_height / 2 + 0.1 - diamond_size),
        (indicator_x - diamond_size, indicator_y + gauge_height / 2 + 0.1)
    ], color=indicator_color, zorder=11)
    ax.add_patch(diamond)

    # Add a small circle at the center of the diamond for additional refinement
    circle = plt.Circle((indicator_x, indicator_y + gauge_height / 2 + 0.1),
                        diamond_size / 2.5, color=IBM_COLORS['white'], zorder=12)
    ax.add_patch(circle)

    # Add score value with IBM styling - make it more prominent
    ax.text(indicator_x, indicator_y + gauge_height / 2 + 0.3, f"{clamped_score:.2f}",
            ha='center', fontsize=18, fontweight='bold', color=indicator_color, zorder=10)

    # Add region labels with IBM styling
    ax.text(-0.7, gauge_y, "Negative", ha='center', va='center', fontsize=14,
            color=neg_text_color, fontweight='medium')
    ax.text(0, gauge_y, "Neutral", ha='center', va='center', fontsize=14,
            color=neu_text_color, fontweight='medium')
    ax.text(0.7, gauge_y, "Positive", ha='center', va='center', fontsize=14,
            color=pos_text_color, fontweight='medium')

    # Determine sentiment description based on score - with IBM terminology
    if clamped_score <= -0.8:
        sentiment_text = "Highly negative"
        sentiment_color = neg_text_color
    elif clamped_score <= -0.3:
        sentiment_text = "Somewhat negative"
        sentiment_color = neg_text_color
    elif clamped_score < -0.1:
        sentiment_text = "Slightly negative"
        sentiment_color = neg_text_color
    elif clamped_score <= 0.1:
        sentiment_text = "Neutral"
        sentiment_color = neu_text_color
    elif clamped_score < 0.3:
        sentiment_text = "Slightly positive"
        sentiment_color = pos_text_color
    elif clamped_score < 0.8:
        sentiment_text = "Somewhat positive"
        sentiment_color = pos_text_color
    else:
        sentiment_text = "Highly positive"
        sentiment_color = pos_text_color

    # Add overall sentiment text with IBM styling - make it more prominent
    ax.text(0, 0.8, f"Overall sentiment: {sentiment_text}", ha='center',
            fontsize=16, fontweight='bold', color=sentiment_color)

    # Set limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.0)

    # Add title with IBM styling
    ax.set_title("Sentiment Analysis", fontsize=18, fontweight='bold',
                 color=IBM_COLORS['black'], pad=20)

    # Add subtle IBM branding
    fig.text(0.01, 0.01, "PDF Analysis AI", fontsize=8, color=IBM_COLORS['gray'])

    return fig, ax

# ---------------------------
# 7. Routing: uploading and parsing PDFs
# ---------------------------

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            tmp_filename = tmp.name

        loader = PyMuPDFLoader(tmp_filename)
        documents = loader.load()
        full_text = "\n".join(doc.page_content for doc in documents if doc.page_content.strip())
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Unable to extract text from PDF")

        return {"pdf_text": full_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# 8. Routing: generating textual reports with IBM styling
# ---------------------------

@app.post("/generate_report")
async def generate_report(request: AnalysisRequest):
    try:
        analysis_type = request.analysis_type.strip()
        valid_types = ["SWOT", "PESTLE", "MOST", "Sentiment"]
        if analysis_type not in valid_types:
            raise HTTPException(status_code=400, detail="The selected analysis type is not valid")

        prompt_template = ANALYSIS_TEMPLATES[analysis_type]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.create_documents([request.pdf_text])

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

        retrieved_docs = vectorstore.similarity_search(prompt_template, k=5)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Enhanced final prompt with clear formatting guidelines
        final_prompt = (
            f"Below is a summary of the contents of the document:\n\n{retrieved_context}\n\n"
            f"Please analyse based on the above and output the results strictly according to the following requirements:\n\n{prompt_template}\n\n"
            f"Important Reminder:\n"
            f"1. Please do not add additional explanations, introductions or summaries\n"
            f"2. Please strictly follow the provided header format and hierarchical structure of the outputs\n"
            f"3. Each point of analysis must be derived directly from the content of the document, without adding extraneous content.\n"
            f"4. All list items must be prefixed with a dash '-'\n"
            f"5. Analyses should be succinct, with no more than 50 words per point"
        )

        # Lowered temperature for more consistent formatting
        response = ollama.chat(
            model="granite3.2",
            messages=[{"role": "user", "content": final_prompt}],
            options={"temperature": 0.2}
        )

        if "message" not in response:
            raise HTTPException(status_code=500, detail="LLM did not return content")

        analysis_text = response["message"]["content"].strip()

        # Helper function: check if analysis meets expected format
        def check_format(analysis_type, text):
            """Check if analysis result meets expected format"""
            if analysis_type == "SWOT":
                return all(x in text for x in ["## Strengths", "## Weaknesses", "## Opportunities", "## Threats"])
            elif analysis_type == "PESTLE":
                return all(x in text for x in
                           ["## Political", "## Economic", "## Social", "## Technological", "## Legal",
                            "## Environmental"])
            elif analysis_type == "MOST":
                return all(x in text for x in ["## Mission", "## Objectives", "## Strategies", "## Tactics"])
            elif analysis_type == "Sentiment":
                return all(x in text for x in
                           ["## Overall sentimental tendencies", "## Positive", "## Neutral", "## Negative",
                            "## Key Emotional Expressions"])
            return False

        # Helper function: check if each section has at least one bullet point
        def check_bullet_points(text):
            sections = text.split("##")[1:]  # Skip first potentially empty part
            for section in sections:
                if section.strip() and "- " not in section:
                    return False
            return True

        # Helper function: generate retry prompt for specific analysis type
        def generate_retry_prompt(analysis_type, text):
            # Create a general retry prefix
            retry_prefix = "Your response did not follow the required format exactly. Please re-export strictly in the following format, keeping all the bullet points but correcting the formatting issues:\n\n"

            if analysis_type == "SWOT":
                missing_sections = []
                for section in ["## Strengths", "## Weaknesses", "## Opportunities", "## Threats"]:
                    if section not in text:
                        missing_sections.append(section)

                retry_prompt = retry_prefix
                if missing_sections:
                    retry_prompt += f"Missing sections: {', '.join(missing_sections)}\n\n"

                retry_prompt += (
                    "A complete SWOT analysis must contain the following four sections, with at least three bullet points in each section:\n\n"
                    "## Strengths\n- [Strength 1]\n- [Strength 2]\n- [Strength 3]\n\n"
                    "## Weaknesses\n- [Weakness 1]\n- [Weakness 2]\n- [Weakness 3]\n\n"
                    "## Opportunities\n- [Opportunity 1]\n- [Opportunity 2]\n- [Opportunity 3]\n\n"
                    "## Threats\n- [Threat 1]\n- [Threat 2]\n- [Threat 3]\n\n"
                    "Make sure:\n"
                    "1. Use a dash '-' before all points\n"
                    "2. Maintain the precise title format (including ### and spaces) as described above\n"
                    "3. Retain the content points you analysed earlier and only adjust the formatting"
                )

            elif analysis_type == "PESTLE":
                missing_sections = []
                for section in ["## Political", "## Economic", "## Social", "## Technological", "## Legal",
                                "## Environmental"]:
                    if section not in text:
                        missing_sections.append(section)

                retry_prompt = retry_prefix
                if missing_sections:
                    retry_prompt += f"The following sections are missing: {', '.join(missing_sections)}\n\n"

                retry_prompt += (
                    "A complete PESTLE analysis must contain the following six sections with at least three bullet points in each section:\n\n"
                    "## Political\n- [Political factor 1]\n- [Political factor 2]\n- [Political factor 3]\n\n"
                    "## Economic\n- [Economic factor 1]\n- [Economic factor 2]\n- [Economic factor 3]\n\n"
                    "## Social\n- [Social factor 1]\n- [Social factor 2]\n- [Social factor 3]\n\n"
                    "## Technological\n- [Technological factor 1]\n- [Technological factor 2]\n- [Technological factor 3]\n\n"
                    "## Legal\n- [Legal factor 1]\n- [Legal factor 2]\n- [Legal factor 3]\n\n"
                    "## Environmental\n- [Environmental factor 1]\n- [Environmental factor 2]\n- [Environmental factor 3]\n\n"
                    "Make sure:\n"
                    "1. Use a dash '-' before all points\n"
                    "2. Maintain the precise title format (including ### and spaces) as described above\n"
                    "3. Retain the content points you analysed earlier and only adjust the formatting"
                )

            elif analysis_type == "MOST":
                missing_sections = []
                for section in ["## Mission", "## Objectives", "## Strategies", "## Tactics"]:
                    if section not in text:
                        missing_sections.append(section)

                retry_prompt = retry_prefix
                if missing_sections:
                    retry_prompt += f"The following sections are missing: {', '.join(missing_sections)}\n\n"

                retry_prompt += (
                    "A complete MOST analysis must contain the following four sections with sufficient bullet points for each section:\n\n"
                    "## Mission\n- [Mission]\n\n"
                    "## Objectives\n- [Objective 1]\n- [Objective 2]\n- [Objective 3]\n\n"
                    "## Strategies\n- [Strategy 1]\n- [Strategy 2]\n- [Strategy 3]\n\n"
                    "## Tactics\n- [Tactic 1]\n- [Tactic 2]\n- [Tactic 3]\n- [Tactic 4]\n\n"
                    "Make sure:\n"
                    "1. Use a dash '-' before all points\n"
                    "2. The mission section should identify the 'core purpose' and 'long-term vision' of the organization. Summarize it in a 'clear and concise' statement\n"
                    "3. Each section has at least the minimum number of points shown above\n"
                    "4. Keep the exact title format (including ### and spaces) as described above\n"
                    "5. Retain the content points you analysed earlier and only adjust the formatting"
                )

            elif analysis_type == "Sentiment":
                missing_sections = []
                for section in ["## Overall sentimental tendencies", "## Positive", "## Neutral", "## Negative",
                                "## Key Emotional Expressions"]:
                    if section not in text:
                        missing_sections.append(section)

                retry_prompt = retry_prefix
                if missing_sections:
                    retry_prompt += f"The following sections are missing: {', '.join(missing_sections)}\n\n"

                retry_prompt += (
                    "A complete sentiment analysis must contain the following five components:\n\n"
                    "## Overall sentimental Tendency\n- Emotional Score: [a value between -1 and 1]\n- Overall Rating: [short description ofOverall sentimental tendency]\n\n"
                    "## Positive\n- [Positive Emotional Description 1]\n- [Positive Emotional Description 2]\n\n"
                    "## Neutral\n- [Neutral Emotional Description 1]\n- [Neutral Emotional Description 2]\n\n"
                    "## Negative\n- [Negative Emotional Description 1]\n- [Negative Emotional Description 2]\n\n"
                    "## Key Emotional Expressions\n1. \"[Key Emotional Sentences 1]\"\n2. \"[Key Emotional Sentences 2]\"\n3. \"[Key Emotional Sentences 3]\"\n\n"
                    "Make sure:\n"
                    "1. The overall affective tendency must contain a rating between -1 and 1.\n"
                    "2. The main points of the sections on types of emotions must begin with a dash '-'.\n"
                    "3. Expressions of key emotions must be numbered and enclose sentences in inverted commas\n"
                    "4. Maintain the precise title format (including ## and spaces) as described above\n"
                    "5. Retain the content points you analysed earlier and adjust the formatting only."
                )

            else:
                retry_prompt = "Please regenerate the analysis in the requested format."

            return retry_prompt

        # Helper function: try to extract content from incorrectly formatted text and restructure
        def attempt_restructure(analysis_type, text):
            """Try to extract content from incorrectly formatted text and reorganize into correct format"""
            if analysis_type == "SWOT":
                # Identify possible SWOT sections
                strengths = []
                weaknesses = []
                opportunities = []
                threats = []

                # Extract content for each section
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # Identify section
                    if "strength" in lower_line or "优势" in lower_line:
                        current_section = "strengths"
                        continue
                    elif "weakness" in lower_line or "劣势" in lower_line:
                        current_section = "weaknesses"
                        continue
                    elif "opportunit" in lower_line or "机会" in lower_line:
                        current_section = "opportunities"
                        continue
                    elif "threat" in lower_line or "威胁" in lower_line:
                        current_section = "threats"
                        continue

                    # Collect points
                    if current_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        if current_section == "strengths":
                            strengths.append(line)
                        elif current_section == "weaknesses":
                            weaknesses.append(line)
                        elif current_section == "opportunities":
                            opportunities.append(line)
                        elif current_section == "threats":
                            threats.append(line)

                # Ensure each section has at least one point
                if not strengths:
                    strengths = ["- Failure to identify clear strengths"]
                if not weaknesses:
                    weaknesses = ["- Failure to identify clear weaknesses"]
                if not opportunities:
                    opportunities = ["- Failure to identify clear opportunities"]
                if not threats:
                    threats = ["- Failure to identify clear threats"]

                # Restructure into correct format
                restructured = (
                        "## Strengths\n" + "\n".join(strengths) + "\n\n" +
                        "## Weaknesses\n" + "\n".join(weaknesses) + "\n\n" +
                        "## Opportunities\n" + "\n".join(opportunities) + "\n\n" +
                        "## Threats\n" + "\n".join(threats)
                )

                return restructured

            elif analysis_type == "PESTLE":
                # Identify possible PESTLE sections
                political = []
                economic = []
                social = []
                technological = []
                legal = []
                environmental = []

                # Extract content for each section
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # Identify section
                    if any(x in lower_line for x in ["politic", "政治"]):
                        current_section = "political"
                        continue
                    elif any(x in lower_line for x in ["econom", "经济"]):
                        current_section = "economic"
                        continue
                    elif any(x in lower_line for x in ["social", "社会"]):
                        current_section = "social"
                        continue
                    elif any(x in lower_line for x in ["tech", "技术"]):
                        current_section = "technological"
                        continue
                    elif any(x in lower_line for x in ["legal", "法律"]):
                        current_section = "legal"
                        continue
                    elif any(x in lower_line for x in ["environ", "环境"]):
                        current_section = "environmental"
                        continue

                    # Collect points
                    if current_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        if current_section == "political":
                            political.append(line)
                        elif current_section == "economic":
                            economic.append(line)
                        elif current_section == "social":
                            social.append(line)
                        elif current_section == "technological":
                            technological.append(line)
                        elif current_section == "legal":
                            legal.append(line)
                        elif current_section == "environmental":
                            environmental.append(line)

                # Ensure each section has at least one point
                if not political:
                    political = ["- No clear political factors identified"]
                if not economic:
                    economic = ["- No clear economic factors identified"]
                if not social:
                    social = ["- No clear social factors identified"]
                if not technological:
                    technological = ["- No clear technological factors identified"]
                if not legal:
                    legal = ["- No clear legal factors identified"]
                if not environmental:
                    environmental = ["- No clear environmental factors identified"]

                # Restructure into correct format
                restructured = (
                        "## Political\n" + "\n".join(political) + "\n\n" +
                        "## Economic\n" + "\n".join(economic) + "\n\n" +
                        "## Social\n" + "\n".join(social) + "\n\n" +
                        "## Technological\n" + "\n".join(technological) + "\n\n" +
                        "## Legal\n" + "\n".join(legal) + "\n\n" +
                        "## Environmental\n" + "\n".join(environmental)
                )

                return restructured

            elif analysis_type == "MOST":
                # Identify possible MOST sections
                mission = []
                objectives = []
                strategies = []
                tactics = []

                # Extract content for each section
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # Identify section
                    if any(x in lower_line for x in ["mission", "使命", "任务"]):
                        current_section = "mission"
                        continue
                    elif any(x in lower_line for x in ["objective", "目标"]):
                        current_section = "objectives"
                        continue
                    elif any(x in lower_line for x in ["strateg", "策略"]):
                        current_section = "strategies"
                        continue
                    elif any(x in lower_line for x in ["tactic", "战术", "行动"]):
                        current_section = "tactics"
                        continue

                    # Collect points
                    if current_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        if current_section == "mission":
                            mission.append(line)
                        elif current_section == "objectives":
                            objectives.append(line)
                        elif current_section == "strategies":
                            strategies.append(line)
                        elif current_section == "tactics":
                            tactics.append(line)

                # Ensure each section has at least one point
                if not mission:
                    mission = ["- No clear mission statement identified"]
                if not objectives:
                    objectives = ["- No clear objectives identified"]
                if not strategies:
                    strategies = ["- No clear strategies identified"]
                if not tactics:
                    tactics = ["- No clear tactics identified"]

                # Restructure into correct format
                restructured = (
                        "## Mission\n" + "\n".join(mission) + "\n\n" +
                        "## Objectives\n" + "\n".join(objectives) + "\n\n" +
                        "## Strategies\n" + "\n".join(strategies) + "\n\n" +
                        "## Tactics\n" + "\n".join(tactics)
                )

                return restructured

            elif analysis_type == "Sentiment":
                # Identify possible Sentiment sections
                overall = []
                positive = []
                neutral = []
                negative = []
                expressions = []

                # Extract sentiment score
                sentiment_score = 0
                score_pattern = r'-?\d+\.?\d*'
                score_matches = re.findall(score_pattern, text)
                if score_matches:
                    try:
                        potential_scores = [float(s) for s in score_matches if -1 <= float(s) <= 1]
                        if potential_scores:
                            sentiment_score = potential_scores[0]
                    except:
                        pass

                # Extract content for each section
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # Identify section
                    if "overall" in lower_line or "tendency" in lower_line or "tendencies" in lower_line:
                        current_section = "overall"
                        continue
                    elif "positive" in lower_line:
                        current_section = "positive"
                        continue
                    elif "neutral" in lower_line:
                        current_section = "neutral"
                        continue
                    elif "negative" in lower_line:
                        current_section = "negative"
                        continue
                    elif "key" in lower_line and ("expression" in lower_line or "emotion" in lower_line):
                        current_section = "expressions"
                        continue

                    # Collect points
                    if current_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        if current_section == "overall":
                            overall.append(line)
                        elif current_section == "positive":
                            positive.append(line)
                        elif current_section == "neutral":
                            neutral.append(line)
                        elif current_section == "negative":
                            negative.append(line)
                    elif current_section == "expressions" and (
                            line.startswith('"') or line.startswith('"') or line.startswith("'")):
                        expressions.append(line)
                    elif current_section == "expressions" and (line[0].isdigit() and line[1] in ('.', ')', ' ')):
                        expressions.append(line)

                # Ensure sentiment score is included in overall section
                if not any("score" in item.lower() for item in overall):
                    overall.insert(0, f"- Sentiment Score: {sentiment_score}")

                if not any("rating" in item.lower() for item in overall):
                    if sentiment_score > 0.3:
                        overall.append("- Overall Rating: The text shows generally positive sentiment.")
                    elif sentiment_score < -0.3:
                        overall.append("- Overall Rating: The text shows generally negative sentiment.")
                    else:
                        overall.append("- Overall Rating: The text shows generally neutral sentiment.")

                # Ensure each section has at least two points
                if len(positive) < 2:
                    positive.append("- Positive elements not clearly identified")
                if len(neutral) < 2:
                    neutral.append("- Neutral elements not clearly identified")
                if len(negative) < 2:
                    negative.append("- Negative elements not clearly identified")

                # Format expressions as numbered list
                formatted_expressions = []
                for i, expr in enumerate(expressions, 1):
                    if expr.startswith('"') or expr.startswith('"') or expr.startswith("'"):
                        formatted_expressions.append(f"{i}. {expr}")
                    else:
                        # Extract the quote part
                        match = re.search(r'[""](.+?)[""]', expr)
                        if match:
                            formatted_expressions.append(f"{i}. \"{match.group(1)}\"")
                        else:
                            # Just convert to numbered format if no quotes found
                            text_part = expr[2:].strip() if expr.startswith('-') else expr
                            formatted_expressions.append(f"{i}. \"{text_part}\"")

                # Add default expressions if none found
                if not formatted_expressions:
                    formatted_expressions = [
                        '1. "No key emotional expressions identified"',
                        '2. "Analysis could not extract specific quoted content"'
                    ]

                # Restructure into correct format
                restructured = (
                        "## Overall sentimental tendencies\n" + "\n".join(overall) + "\n\n" +
                        "## Positive emotion\n" + "\n".join(positive) + "\n\n" +
                        "## Neutral emotion\n" + "\n".join(neutral) + "\n\n" +
                        "## Negative emotion\n" + "\n".join(negative) + "\n\n" +
                        "## Key Emotional Expressions\n" + "\n".join(formatted_expressions)
                )

                return restructured

            # If none of the structured types match or restructuring fails, return the original text
            return text

        # Maximum retry attempts
        max_retries = 3
        current_retry = 0

        while current_retry < max_retries:
            # Check format correctness
            format_correct = check_format(analysis_type, analysis_text)
            bullet_points_correct = check_bullet_points(analysis_text)

            if format_correct and bullet_points_correct:
                break  # Format is correct, no need to retry

            # Generate retry prompt
            retry_prompt = generate_retry_prompt(analysis_type, analysis_text)

            # Log retry info
            print(
                f"Retry #{current_retry + 1}: {'Formatting error' if not format_correct else ''} {'Missing list items' if not bullet_points_correct else ''}")

            # Lower temperature with each retry
            retry_temperature = max(0.1, 0.3 - current_retry * 0.1)

            retry_response = ollama.chat(
                model="granite3.2",
                messages=[
                    {"role": "user", "content": final_prompt},
                    {"role": "assistant", "content": analysis_text},
                    {"role": "user", "content": retry_prompt}
                ],
                options={"temperature": retry_temperature}
            )

            if "message" in retry_response:
                new_text = retry_response["message"]["content"].strip()

                # Check if new text has better format
                new_format_correct = check_format(analysis_type, new_text)
                new_bullet_points_correct = check_bullet_points(new_text)

                # Only replace if new text has better format
                if (new_format_correct and not format_correct) or (
                        new_format_correct == format_correct and new_bullet_points_correct and not bullet_points_correct):
                    analysis_text = new_text
                elif not new_format_correct and not format_correct:
                    # Try to restructure original text
                    restructured_text = attempt_restructure(analysis_type, analysis_text)
                    if check_format(analysis_type, restructured_text):
                        analysis_text = restructured_text

            current_retry += 1

        # Final emergency fix attempt
        if not check_format(analysis_type, analysis_text):
            emergency_text = attempt_restructure(analysis_type, analysis_text)
            if check_format(analysis_type, emergency_text):
                analysis_text = emergency_text

        # Generate PDF report with IBM styling
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create IBM-styled paragraph styles
        ibm_title_style = ParagraphStyle(
            'IBMTitle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor(IBM_COLORS['black']),
            spaceAfter=12,
            spaceBefore=12,
            alignment=TA_CENTER
        )

        ibm_heading_style = ParagraphStyle(
            'IBMHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor(IBM_COLORS['blue']),
            spaceAfter=8,
            spaceBefore=16
        )

        ibm_bullet_style = ParagraphStyle(
            'IBMBullet',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor(IBM_COLORS['black']),
            leftIndent=20,
            firstLineIndent=-15,
            spaceBefore=2,
            spaceAfter=5
        )

        # Prepare story elements
        elements = []

        # Add title
        title = Paragraph(f"{analysis_type} Analysis Report", ibm_title_style)
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Process analysis text into well-formatted paragraphs
        lines = analysis_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("## "):
                # Section heading
                current_section = line[3:].strip()
                heading = Paragraph(current_section, ibm_heading_style)
                elements.append(heading)
            elif line.startswith("- ") or line.startswith("• "):
                # Convert to IBM bullet style
                bullet_text = line[2:].strip()
                bullet = Paragraph(f"• {bullet_text}", ibm_bullet_style)
                elements.append(bullet)
            elif line[0].isdigit() and line[1:3] in [". ", ") "]:
                # Numbered item, convert to IBM style
                numbered_text = line[3:].strip()
                if "\"" in numbered_text:
                    # Keep original numbering for quoted content
                    numbered = Paragraph(f"{line[0]}. {numbered_text}", ibm_bullet_style)
                else:
                    # Convert numbers to bullets for consistency
                    numbered = Paragraph(f"• {numbered_text}", ibm_bullet_style)
                elements.append(numbered)
            else:
                # Normal text paragraph
                normal = Paragraph(line, styles['Normal'])
                elements.append(normal)

        # Build PDF document
        doc.build(elements)
        pdf_buffer.seek(0)

        return JSONResponse(
            content={
                "text": analysis_text,
                "pdf": base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# 9. Routing: generating word clouds
# ---------------------------

@app.post("/generate_wordcloud")
async def generate_wordcloud_api(request: AnalysisRequest):
    try:
        text = request.pdf_text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text content is empty, can't generate word cloud map")

        # Create a fallback wordcloud in case of errors
        def create_fallback_wordcloud():
            """Create a simple fallback wordcloud with error message"""
            plt = set_ibm_matplotlib_style()
            fig, ax = plt.subplots(figsize=(10, 6), facecolor=IBM_COLORS['white'])
            ax.text(0.5, 0.5, "Unable to generate word cloud from this text.",
                    ha='center', va='center', fontsize=14, color=IBM_COLORS['red'])
            ax.text(0.5, 0.4, "The text may be too short or contain unsupported content.",
                    ha='center', va='center', fontsize=12, color=IBM_COLORS['gray_80'])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Word Cloud Error", fontsize=16, fontweight='bold', color=IBM_COLORS['black'])
            ax.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=200, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf

        try:
            wc_buffer = generate_wordcloud_image(text)
        except Exception as wc_error:
            # Log error and create fallback
            print(f"WordCloud generation error: {str(wc_error)}")
            import traceback
            print(traceback.format_exc())
            wc_buffer = create_fallback_wordcloud()

        wordcloud_base64 = base64.b64encode(wc_buffer.getvalue()).decode("utf-8")
        return JSONResponse(content={"wordcloud": wordcloud_base64})

    except Exception as e:
        # Main error handler
        import traceback
        print(f"WordCloud endpoint error: {str(e)}")
        print(traceback.format_exc())

        # Return an error image instead of failing
        plt = set_ibm_matplotlib_style()
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=IBM_COLORS['white'])
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center',
                fontsize=14, color=IBM_COLORS['red'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        plt.close(fig)

        error_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return JSONResponse(content={"wordcloud": error_base64})


# ---------------------------
# 10. Routing: four-quadrant or histogram visualization
# ---------------------------

@app.post("/generate_visualization")
async def generate_visualization(request: VisualizationRequest):
    try:
        # Case-insensitive processing of analysis type
        raw_type = request.analysis_type
        analysis_type = raw_type.lower().strip()
        text = request.analysis_text.strip() if request.analysis_text else ""

        if not text:
            raise HTTPException(status_code=400,
                                detail="The analysis text is empty and no visualization can be generated")

        if "swot" in analysis_type:
            # Parse SWOT data and generate IBM-styled visualization
            sections = parse_swot_analysis(text)
            fig, ax = generate_swot_visualization(sections)

        elif "pestle" in analysis_type:
            # Parse PESTLE data and generate IBM-styled visualization
            sections = parse_pestle_analysis(text)
            fig, ax = generate_pestle_visualization(sections)

        elif "most" in analysis_type:
            # Parse MOST data and generate IBM-styled visualization
            sections = parse_most_analysis(text)
            fig, ax = generate_most_visualization(sections)
            # Set appropriate figure size with IBM styling
            fig.set_size_inches(14, 10)
            fig.tight_layout(pad=2.0)

        elif "sentiment" in analysis_type:
            # Extract sentiment score and generate IBM-styled visualization
            score = parse_sentiment_analysis(text)
            fig, ax = generate_sentiment_visualization(score)

        else:
            raise HTTPException(status_code=400, detail="Analysis type not supported")

        # Output as image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return JSONResponse(content={"chart": chart_base64})
    except Exception as e:
        # Provide detailed error for debugging
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log detailed error
        raise HTTPException(status_code=500, detail=f"Visualization generation failure: {str(e)}")



if __name__ == "__main__":
    import uvicorn

    # Start FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)

