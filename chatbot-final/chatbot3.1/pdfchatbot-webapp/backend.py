from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import ollama
import io
import base64
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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

app = FastAPI()

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
        "- [Political factors 2\n"
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
# 2. Auxiliary functions
# ---------------------------

def generate_wordcloud_image(text: str):
    """
    Generate a word cloud from text with enhanced stopwords for financial reports,
    including months, and converting all words to lowercase manually
    """
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

    # Create the word cloud (without the unsupported lowercase parameter)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        collocations=False,  # Don't include bigrams like "annual report"
        max_words=100,  # Limit to 100 most common words
        min_word_length=3  # Skip very short words
    ).generate(text)

    # Convert to image
    img_buffer = io.BytesIO()
    wc.to_image().save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return img_buffer


# 修改 add_wrapped_text 函数，确保正确处理比较操作
def add_wrapped_text(ax, text, x, y, width, fontsize=10, max_lines=5):
    """更改后的函数，更好地处理文本换行和显示"""
    # 确保文本不为None
    if text is None or text == "":
        text = "(No data)"

    # 确保参数是浮点数
    x = float(x)
    y = float(y)
    width = float(width)
    fontsize = float(fontsize)

    # 标准化文本开头，保留项目符号
    if text.startswith('-') or text.startswith('•'):
        text = text.strip()
    else:
        text = '• ' + text.strip()

    # 调整包装宽度，使其更合理
    # 对于短文本增加宽度以容纳更多字符
    wrap_width = int(70 * width)  # 这是一个经验值，可能需要调整

    # 使用textwrap包装文本
    wrapped_lines = textwrap.wrap(text, width=wrap_width)

    # 检查空行情况
    if not wrapped_lines:
        wrapped_lines = ["(No content)"]

    # 限制行数并添加省略号
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines - 1]
        wrapped_lines.append("...")

    # 绘制每一行文本
    for i, line in enumerate(wrapped_lines):
        ax.text(x, y - i * (fontsize * 0.2), line, fontsize=fontsize,
                horizontalalignment='left', verticalalignment='top',
                wrap=True)

    # 返回下一行的y位置，增加额外间距
    return y - len(wrapped_lines) * (fontsize * 0.2) - 0.1  # 增加额外间距


def parse_swot_analysis(text):
    # 防止None值
    if text is None:
        text = ""

    sections = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
    current_section = None

    # 将文本分割为行
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查部分标题
        lower_line = line.lower()
        if '##' in line or '#' in line:
            if ('优势' in lower_line) or ('strength' in lower_line):
                current_section = 'strengths'
                continue
            elif ('劣势' in lower_line) or ('weakness' in lower_line):
                current_section = 'weaknesses'
                continue
            elif ('机会' in lower_line) or ('opportunit' in lower_line):
                current_section = 'opportunities'
                continue
            elif ('威胁' in lower_line) or ('threat' in lower_line):
                current_section = 'threats'
                continue

        # 如果我们在某个部分中且该行以项目符号开始，则添加它
        if current_section and (line.startswith('-') or line.startswith('•')):
            sections[current_section].append(line)
        # 如果我们在某个部分中但没有项目符号，尝试从内容中识别部分
        elif current_section:
            # 非项目符号但在当前部分中，作为普通文本添加
            if not ('##' in line or '#' in line):
                sections[current_section].append('• ' + line)
        # 尝试从没有明确标记的文本中识别部分
        elif ('优势' in lower_line) or ('strength' in lower_line):
            sections['strengths'].append('• ' + line)
        elif ('劣势' in lower_line) or ('weakness' in lower_line):
            sections['weaknesses'].append('• ' + line)
        elif ('机会' in lower_line) or ('opportunit' in lower_line):
            sections['opportunities'].append('• ' + line)
        elif ('威胁' in lower_line) or ('threat' in lower_line):
            sections['threats'].append('• ' + line)

    return sections


def generate_swot_visualization(sections):
    """
    Creates a SWOT quadrant diagram with improved text density and spacing
    """
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(14, 10))

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw cross
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=1.5)

    # Configure axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    ax.set_title("SWOT Analysis Quadrant Diagram", fontsize=16, fontweight='bold', pad=20)

    # Add background colors to quadrants
    ax.add_patch(plt.Rectangle((-1, 0), 1, 1, fill=True, color='#e6ffe6', alpha=0.3))  # Strengths
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='#ffe6e6', alpha=0.3))  # Weaknesses
    ax.add_patch(plt.Rectangle((-1, -1), 1, 1, fill=True, color='#e6e6ff', alpha=0.3))  # Opportunities
    ax.add_patch(plt.Rectangle((0, -1), 1, 1, fill=True, color='#f9e6ff', alpha=0.3))  # Threats

    # Add quadrant titles
    ax.text(-0.95, 0.95, "Strengths", fontsize=14, fontweight='bold', color='green')
    ax.text(0.05, 0.95, "Weaknesses", fontsize=14, fontweight='bold', color='red')
    ax.text(-0.95, -0.05, "Opportunities", fontsize=14, fontweight='bold', color='blue')
    ax.text(0.05, -0.05, "Threats", fontsize=14, fontweight='bold', color='purple')

    # Function to add text with improved word wrapping and increased chars per line
    def add_word_wrapped_text(ax, text, x, y, fontsize=10, line_spacing=0.05):
        if not text or not isinstance(text, str):
            return y

        # Clean up text
        if text.startswith('-') or text.startswith('•'):
            clean_text = text[1:].strip()
            bullet = "• "
        else:
            clean_text = text.strip()
            bullet = "• "

        # Add bullet only for first line
        ax.text(x, y, bullet, fontsize=fontsize, ha='left', va='top')

        # Split text into words
        words = clean_text.split()
        
        # Calculate chars per line - INCREASED to allow more text per line
        # Using a higher value for more text density
        chars_per_quadrant = 65  # Increased from previous ~45 chars
        
        # Process text to fit within quadrant
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Add word length plus a space
            word_length = len(word) + 1
            
            if current_length + word_length <= chars_per_quadrant:
                current_line.append(word)
                current_length += word_length
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
        
        # Add the last line if not empty
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw text lines
        current_y = y
        text_x = x + 0.03  # Indent text after bullet
        
        for line in lines:
            ax.text(text_x, current_y, line, fontsize=fontsize, ha='left', va='top')
            current_y -= line_spacing
        
        # Return final y position with extra spacing
        return current_y - 0.03

    # Function to add multiple bullet points with dynamic spacing
    def add_bullet_list(items, x, y, quadrant_height=0.9, fontsize=10):
        current_y = y
        
        for item in items:
            if item and isinstance(item, str):
                # Estimate lines based on text length with new chars_per_line
                approx_lines = max(1, len(item) // 65)  # Adjusted to match new chars_per_line
                spacing_needed = approx_lines * 0.05 + 0.03
                
                # Check if we have enough space
                if current_y - spacing_needed > (y - quadrant_height):
                    current_y = add_word_wrapped_text(ax, item, x, current_y, fontsize=fontsize)
                else:
                    # Add ellipsis to indicate more content
                    ax.text(x+0.03, current_y, "...", fontsize=fontsize, ha='left', va='top')
                    break

    # Add content to each quadrant with adjusted parameters for better text density
    # Using slightly larger font size now that we have better text wrapping
    add_bullet_list(sections["strengths"], -0.95, 0.85, fontsize=10)
    add_bullet_list(sections["weaknesses"], 0.05, 0.85, fontsize=10)
    add_bullet_list(sections["opportunities"], -0.95, -0.15, fontsize=10)
    add_bullet_list(sections["threats"], 0.05, -0.15, fontsize=10)

    return fig, ax

def parse_pestle_analysis(text):
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

    # 分割文本为行
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查部分标题
        lower_line = line.lower()

        # 识别各个部分
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
            # 如果行中包含关键词，也可以识别部分
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

        # 收集当前部分的内容
        if current_section and line:
            # 如果是列表项，保留原样；否则添加项目符号
            if line.startswith('-') or line.startswith('•'):
                sections[current_section].append(line)
            elif line[0].isdigit() and line[1] in ['.', ')', ' ']:
                # 将数字编号格式转换为短横线格式
                sections[current_section].append('- ' + line[2:].strip())
            elif not ('##' in line or '#' in line):
                # 不是标题行的普通文本也加入
                sections[current_section].append('• ' + line)

    return sections


def generate_pestle_visualization(sections):
    """
    Creates a PESTLE analysis visualization with improved text wrapping,
    correct top-to-bottom order, and significantly increased line spacing
    to prevent any text overlap
    """
    # Set up constants
    words_per_line = 12  # 12 words per line as requested
    line_spacing = 0.2  # Significantly increased from 0.1 to 0.2 for better line separation
    item_spacing = 0.3  # Significantly increased from 0.15 to 0.3 for better bullet point separation

    # Define PESTLE components in correct top-to-bottom order and their colors
    pestle_data = [
        {"letter": "P", "name": "Political", "color": "#4a6fe3",
         "insights": sections["political"][:3] if sections["political"] else ["No political factors provided"]},
        {"letter": "E", "name": "Economic", "color": "#e74c3c",
         "insights": sections["economic"][:3] if sections["economic"] else ["No economic factors provided"]},
        {"letter": "S", "name": "Social", "color": "#f39c12",
         "insights": sections["social"][:3] if sections["social"] else ["No social factors provided"]},
        {"letter": "T", "name": "Technological", "color": "#e2cb92",
         "insights": sections["technological"][:3] if sections["technological"] else ["No technological factors provided"]},
        {"letter": "L", "name": "Legal", "color": "#8ecfd5",
         "insights": sections["legal"][:3] if sections["legal"] else ["No legal factors provided"]},
        {"letter": "E", "name": "Environmental", "color": "#aed581",
         "insights": sections["environmental"][:3] if sections["environmental"] else ["No environmental factors provided"]}
    ]

    # Reverse the order for top-to-bottom display
    pestle_data.reverse()

    # Pre-calculate row heights with much more space
    row_heights = []
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

    # Calculate total height needed
    total_height = sum(row_heights)

    # Create figure with calculated dimensions and increased height
    fig, ax = plt.subplots(figsize=(14, total_height * 1.5))  # Further increased height multiplier

    # Turn off axes
    ax.axis('off')

    # Function to wrap and display text with significantly increased line spacing
    def add_word_limited_text(text, x_start, y_start, fontsize=10):
        if not text or not isinstance(text, str):
            return y_start

        # Clean the text
        if text.startswith('-') or text.startswith('•'):
            clean_text = text[1:].strip()
        else:
            clean_text = text.strip()

        # Add bullet point
        ax.text(x_start, y_start, "•", fontsize=fontsize, ha='left', va='top')

        # Split into words and group into lines with more spacing
        words = clean_text.split()
        text_x = x_start + 0.08  # Further increased indent after bullet
        current_y = y_start

        for i in range(0, len(words), words_per_line):
            line_words = words[i:i + words_per_line]
            line = " ".join(line_words)
            ax.text(text_x, current_y, line, fontsize=fontsize, ha='left', va='top')
            current_y -= line_spacing  # Using the increased line spacing

        # Return the new y position with extra space for the next item
        return current_y - item_spacing

    # Create PESTLE grid with dynamic heights and more spacing
    current_y = 0  # Start from the top

    for i, (item, row_height) in enumerate(zip(pestle_data, row_heights)):
        # Calculate row positions
        row_top = current_y + row_height
        row_bottom = current_y

        # Letter box (left column)
        ax.add_patch(plt.Rectangle((0, row_bottom), 1, row_height, fill=True, color=item["color"]))
        ax.text(0.5, row_bottom + row_height / 2, item["letter"], ha='center', va='center', color='white',
                fontsize=36, fontweight='bold')

        # Name box (second column)
        ax.add_patch(plt.Rectangle((1, row_bottom), 2, row_height, fill=True, color=item["color"], alpha=0.7))
        ax.text(2, row_bottom + row_height / 2, item["name"], ha='center', va='center', color='white',
                fontsize=16, fontweight='bold')

        # Content box (right column)
        ax.add_patch(plt.Rectangle((3, row_bottom), 7, row_height, fill=True, color=item["color"], alpha=0.3))

        # Add insights with much more space between lines
        # Start with more space from the top
        content_y = row_bottom + row_height - 0.3

        for insight in item["insights"][:3]:
            if insight and isinstance(insight, str):
                content_y = add_word_limited_text(insight, 3.2, content_y)

        # Move to the next row
        current_y += row_height

    # Set chart dimensions
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total_height)

    # Set title
    ax.set_title("PESTLE Analysis", fontsize=16, fontweight='bold', pad=20)

    return fig, ax


def parse_most_analysis(text):
    """
    From text parsing MOST analysis of the respective parts

    Parameters:
    text - MOST analysis text

    Return:
    Dictionary containing Mission, Objectives, Strategies, Tactics sections content
    """
    # Prevent None value
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

        # Check section titles
        lower_line = line.lower()

        # Identify each section's start
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
    Creates a MOST analysis visualization with correct pyramid shape:
    - Mission at the top (small tip)
    - Tactics at the bottom (wide base)
    - Fixed absolute positioning of text sections
    - Wider, more visually appealing pyramid
    """
    # Set up constants for text formatting
    words_per_line = 10  # Limit words per line for better readability
    line_spacing = 0.3  # Spacing between lines of the same bullet point
    item_spacing = 0.6  # Spacing between different bullet points

    # Get content for each section
    mission_items = sections["mission"][:2] if sections["mission"] else ["No mission information provided"]
    objectives_items = sections["objectives"][:3] if sections["objectives"] else ["No objective information provided"]
    strategies_items = sections["strategies"][:3] if sections["strategies"] else ["No strategy information provided"]
    tactics_items = sections["tactics"][:4] if sections["tactics"] else ["No tactic information provided"]

    # Set up pyramid dimensions
    pyramid_width = 10  # Wider pyramid
    pyramid_height = 12  # Fixed height

    # Create figure with fixed dimensions
    fig, ax = plt.subplots(figsize=(18, 18))  # Large figure size for better spacing

    # Turn off axis
    ax.axis('off')

    # Define coordinates for a proper pyramid shape
    # The tip (Mission) is at the top, base (Tactics) at the bottom
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
        # For a proper pyramid: width = baseWidth * (1 - height/totalHeight)
        width_at_height = pyramid_width * (1 - height / pyramid_height)
        left_x = (pyramid_width - width_at_height) / 2
        right_x = left_x + width_at_height
        divider_points.append([(left_x, height), (right_x, height)])

    # Define colors for each section
    colors = [
        "#2c6b97",  # Mission (top) - dark blue
        "#e77c4d",  # Objectives - orange
        "#337544",  # Strategies - green
        "#38a9d4"   # Tactics (bottom) - light blue
    ]

    # Create polygon patches for each section
    # Mission section (top tip)
    mission_polygon = plt.Polygon([
        (pyramid_tip_x, pyramid_tip_y),  # Tip
        divider_points[2][0], divider_points[2][1]  # Third divider left
    ], color=colors[0])
    ax.add_patch(mission_polygon)

    # Objectives section
    objectives_polygon = plt.Polygon([
        divider_points[2][0], divider_points[2][1],  # Third divider left
        divider_points[2][1],  # Third divider right
        divider_points[1][1],  # Second divider right
        divider_points[1][0]   # Second divider left
    ], color=colors[1])
    ax.add_patch(objectives_polygon)

    # Strategies section
    strategies_polygon = plt.Polygon([
        divider_points[1][0], divider_points[1][1],  # Second divider
        divider_points[1][1],  # Second divider right
        divider_points[0][1],  # First divider right
        divider_points[0][0]   # First divider left
    ], color=colors[2])
    ax.add_patch(strategies_polygon)

    # Tactics section (bottom base)
    tactics_polygon = plt.Polygon([
        divider_points[0][0], divider_points[0][1],  # First divider
        divider_points[0][1],  # First divider right
        (pyramid_base_right_x, pyramid_base_y),  # Bottom right
        (pyramid_base_left_x, pyramid_base_y)    # Bottom left
    ], color=colors[3])
    ax.add_patch(tactics_polygon)

    # Add section numbers in correct order
    ax.text(pyramid_width / 2, (divider_heights[2] + pyramid_height) / 2, "01", ha='center', va='center',
            color='white', fontsize=20, fontweight='bold')  # Mission - top
    ax.text(pyramid_width / 2, (divider_heights[1] + divider_heights[2]) / 2, "02", ha='center', va='center',
            color='white', fontsize=20, fontweight='bold')  # Objectives
    ax.text(pyramid_width / 2, (divider_heights[0] + divider_heights[1]) / 2, "03", ha='center', va='center',
            color='white', fontsize=20, fontweight='bold')  # Strategies
    ax.text(pyramid_width / 2, divider_heights[0] / 2, "04", ha='center', va='center',
            color='white', fontsize=20, fontweight='bold')  # Tactics - bottom

    # Function to process and display a section with bullet points
    def add_section_with_items(title, items, y_position, content_x, fontsize=11):
        # Add section title
        ax.text(content_x, y_position, title, fontsize=16, fontweight='bold')

        # Start content below title with good spacing
        content_y = y_position - 1.0

        # Add each bullet point
        for item in items:
            if not item or not isinstance(item, str):
                continue

            # Clean text
            if item.startswith('-') or item.startswith('•'):
                clean_text = item[1:].strip()
            else:
                clean_text = item.strip()

            # Add bullet point
            ax.text(content_x, content_y, "•", fontsize=fontsize, ha='left', va='top')

            # Split into words and group into lines
            words = clean_text.split()
            text_x = content_x + 0.5  # Indent after bullet
            current_y = content_y

            # Process each line of this bullet point
            for i in range(0, len(words), words_per_line):
                line_words = words[i:i + words_per_line]
                line = " ".join(line_words)
                ax.text(text_x, current_y, line, fontsize=fontsize, ha='left', va='top')
                current_y -= line_spacing

            # Move to next bullet point with increased spacing
            content_y = current_y - item_spacing

    # Content positioning to the right of pyramid
    content_x = pyramid_width + 1.5

    # Positions for sections - with absolute positioning and large gaps
    # CORRECT ORDER to match the pyramid: Mission at top, Tactics at bottom
    add_section_with_items("Mission", mission_items, pyramid_height + 0.5, content_x)
    add_section_with_items("Objectives", objectives_items, pyramid_height - 4.0, content_x)
    add_section_with_items("Strategies", strategies_items, pyramid_height - 8.5, content_x)
    add_section_with_items("Tactics", tactics_items, pyramid_height - 13.0, content_x)

    # Set chart dimensions
    ax.set_xlim(-1, pyramid_width + 15)
    ax.set_ylim(-2, pyramid_height + 3)

    # Set title
    ax.set_title("MOST Analysis", fontsize=16, fontweight='bold', pad=20)

    return fig, ax

def parse_sentiment_analysis(text):
    """
    Extract sentiment score value from sentiment analysis text, handling cases where
    multiple scores might be present.

    Parameters:
    text - sentiment analysis text

    Returns:
    sentiment score value (-1 to 1 float)
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
    Creates an improved sentiment analysis visualization that accurately
    displays the sentiment score with a better design
    """
    # Create a horizontal gauge figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Turn off axis
    ax.axis('off')

    # Set up gauge regions
    gauge_height = 1.0
    gauge_width = 3.0

    # Create three regions: negative, neutral, positive
    neg_rect = plt.Rectangle((-1.5, -0.5), 1.0, gauge_height, color='#ffcccc', alpha=0.5)
    neu_rect = plt.Rectangle((-0.5, -0.5), 1.0, gauge_height, color='#e6e6e6', alpha=0.5)
    pos_rect = plt.Rectangle((0.5, -0.5), 1.0, gauge_height, color='#ccffcc', alpha=0.5)

    # Add rectangles to chart
    ax.add_patch(neg_rect)
    ax.add_patch(neu_rect)
    ax.add_patch(pos_rect)

    # Add labels for each region
    ax.text(-1.0, -0.2, "Negative", ha='center', va='center', fontsize=16, color='#d9534f')
    ax.text(0.0, -0.2, "Neutral", ha='center', va='center', fontsize=16, color='#777777')
    ax.text(1.0, -0.2, "Positive", ha='center', va='center', fontsize=16, color='#5cb85c')

    # Clamp score between -1 and 1
    clamped_score = max(-1, min(1, score))

    # Draw arrow pointer at the score position
    arrow_y = 0
    ax.arrow(0, arrow_y, clamped_score, 0,
             head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=3)

    # Add score value text
    ax.text(clamped_score, -0.8, f"{clamped_score:.2f}", ha='center', fontsize=18, fontweight='bold')

    # Determine sentiment description based on score
    if clamped_score <= -0.8:
        sentiment_text = "Extremely negative"
    elif clamped_score <= -0.3:
        sentiment_text = "More negative"
    elif clamped_score < -0.1:
        sentiment_text = "Slightly negative"
    elif clamped_score <= 0.1:
        sentiment_text = "Neutral"
    elif clamped_score < 0.3:
        sentiment_text = "Slightly positive"
    elif clamped_score < 0.8:
        sentiment_text = "More positive"
    else:
        sentiment_text = "Extremely positive"

    # Add overall sentiment text
    ax.text(0, 1.0, f"Overall sentiment: {sentiment_text}", ha='center', fontsize=20, fontweight='bold')

    # Set limits
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.5, 1.5)

    # Set title
    ax.set_title("Sentiment analysis results", fontsize=20, pad=20)

    return fig, ax

# ---------------------------
# 3. Routing: uploading and parsing PDFs
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
# 4. Routing: generating textual reports
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

        # 增强最终提示，添加更明确的格式指导
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

        # 增加温度参数，减少随机性，提高格式一致性
        response = ollama.chat(
            model="granite3.2:8b",
            messages=[{"role": "user", "content": final_prompt}],
            options={"temperature": 0.2}  # 使用低温度以提高格式一致性
        )

        if "message" not in response:
            raise HTTPException(status_code=500, detail="LLM did not return content")

        analysis_text = response["message"]["content"].strip()

        # 辅助函数：检查分析结果是否符合预期格式
        def check_format(analysis_type, text):
            """检查分析结果是否符合预期格式"""
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
                           ["## Overall sentimental tendencies", "## Positive", "## Neutral", "## Negative", "## Key Emotional Expressions"])
            return False

        # 辅助函数：检查每个部分是否至少有一个带短横线的列表项
        def check_bullet_points(text):
            sections = text.split("##")[1:]  # 跳过第一个可能为空的部分
            for section in sections:
                if section.strip() and "- " not in section:
                    return False
            return True

        # 辅助函数：生成针对特定分析类型的重试提示
        def generate_retry_prompt(analysis_type, text):
            # 首先创建一个通用的重试前缀
            retry_prefix = "Your response did not follow the required format exactly.Please re-export strictly in the following format, keeping all the bullet points but correcting the formatting issues:\n\n"

            if analysis_type == "SWOT":
                missing_sections = []
                for section in ["## Strengths", "## Weaknesses", "## Opportunities", "## Threats"]:
                    if section not in text:
                        missing_sections.append(section)

                retry_prompt = retry_prefix
                if missing_sections:
                    retry_prompt += f"缺少以下部分: {', '.join(missing_sections)}\n\n"

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
                    "2.The mission section should identify the 'core purpose' and 'long-term vision' of the organization. Summarize it in a 'clear and concise' statement"
                    "3. Each section has at least the minimum number of points shown above\n"
                    "4. Keep the exact title format (including ### and spaces) as described above\n"
                    "5. Retain the content points you analysed earlier and only adjust the formatting"
                )

            elif analysis_type == "Sentiment":
                missing_sections = []
                for section in ["## Overall sentimental tendencies", "## Positive", "## Neutral", "## Negative", "## Key Emotional Expressions"]:
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

        # 辅助函数：尝试从格式错误的文本中提取内容并重新组织
        def attempt_restructure(analysis_type, text):
            """尝试从格式错误的文本中提取内容并重组为正确格式"""

            if analysis_type == "SWOT":
                # 识别可能的SWOT部分
                strengths = []
                weaknesses = []
                opportunities = []
                threats = []

                # 提取各部分内容
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # 识别部分
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

                    # 收集要点
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

                # 确保每个部分至少有一个要点
                if not strengths:
                    strengths = ["- Failure to identify clear strengths"]
                if not weaknesses:
                    weaknesses = ["- Failure to identify clear weaknesses"]
                if not opportunities:
                    opportunities = ["- Failure to identify clear opportunities"]
                if not threats:
                    threats = ["- Failure to identify clear threats"]

                # 重组为正确格式
                restructured = (
                        "## Strengths\n" + "\n".join(strengths) + "\n\n" +
                        "## Weaknesses\n" + "\n".join(weaknesses) + "\n\n" +
                        "## Opportunities\n" + "\n".join(opportunities) + "\n\n" +
                        "## Threats\n" + "\n".join(threats)
                )

                return restructured

            elif analysis_type == "MOST":
                # 识别可能的MOST部分
                mission = []
                objectives = []
                strategies = []
                tactics = []

                # 提取各部分内容
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # 识别部分
                    if "mission" in lower_line or "使命" in lower_line or "任务" in lower_line:
                        current_section = "mission"
                        continue
                    elif "objective" in lower_line or "目标" in lower_line:
                        current_section = "objectives"
                        continue
                    elif "strateg" in lower_line or "策略" in lower_line:
                        current_section = "strategies"
                        continue
                    elif "tactic" in lower_line or "战术" in lower_line or "行动" in lower_line:
                        current_section = "tactics"
                        continue

                    # 收集要点
                    if current_section and (line.startswith('-') or line.startswith('•') or line.startswith('*') or
                                            line[0].isdigit() and line[1] in ['.', ')']):
                        # 统一格式为短横线
                        if line[0].isdigit() and line[1] in ['.', ')']:
                            line = "- " + line[2:].strip()
                        else:
                            line = "- " + line[1:].strip()

                        if current_section == "mission":
                            mission.append(line)
                        elif current_section == "objectives":
                            objectives.append(line)
                        elif current_section == "strategies":
                            strategies.append(line)
                        elif current_section == "tactics":
                            tactics.append(line)

                # 确保每个部分至少有一个要点
                if not mission:
                    mission = ["- Failure to identify a clear mission"]
                if not objectives:
                    objectives = ["- Failure to identify clear objectives", "- Need for further clarification of objectives"]
                if not strategies:
                    strategies = ["- Failed to identify a clear strategy", "- Further strategy development required"]
                if not tactics:
                    tactics = ["- Failure to identify clear tactics", "- Need to further develop specific operational measures"]

                # 重组为正确格式
                restructured = (
                        "## Mission\n" + "\n".join(mission) + "\n\n" +
                        "## Objectives\n" + "\n".join(objectives) + "\n\n" +
                        "## Strategies\n" + "\n".join(strategies) + "\n\n" +
                        "## Tactics\n" + "\n".join(tactics)
                )

                return restructured

            elif analysis_type == "PESTLE":
                # 识别可能的PESTLE部分
                political = []
                economic = []
                social = []
                technological = []
                legal = []
                environmental = []

                # 提取各部分内容
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # 识别部分
                    if "politic" in lower_line or "政治" in lower_line:
                        current_section = "political"
                        continue
                    elif "econom" in lower_line or "经济" in lower_line:
                        current_section = "economic"
                        continue
                    elif "social" in lower_line or "社会" in lower_line:
                        current_section = "social"
                        continue
                    elif "tech" in lower_line or "技术" in lower_line:
                        current_section = "technological"
                        continue
                    elif "legal" in lower_line or "法律" in lower_line:
                        current_section = "legal"
                        continue
                    elif "environ" in lower_line or "环境" in lower_line:
                        current_section = "environmental"
                        continue

                    # 收集要点
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

                # 确保每个部分至少有一个要点
                if not political:
                    political = ["- Failure to identify clear political factors"]
                if not economic:
                    economic = ["- Failure to identify clear economic factors"]
                if not social:
                    social = ["- Failure to identify clear social factors"]
                if not technological:
                    technological = ["- Failure to identify clear technological factors"]
                if not legal:
                    legal = ["- Failure to identify clear legal factors"]
                if not environmental:
                    environmental = ["- Failure to identify clear environmental factors"]

                # 重组为正确格式
                restructured = (
                        "## Political\n" + "\n".join(political) + "\n\n" +
                        "## Economic\n" + "\n".join(economic) + "\n\n" +
                        "## Social\n" + "\n".join(social) + "\n\n" +
                        "## Technological\n" + "\n".join(technological) + "\n\n" +
                        "## Legal\n" + "\n".join(legal) + "\n\n" +
                        "## Environmental\n" + "\n".join(environmental)
                )

                return restructured

            elif analysis_type == "Sentiment":
                # 识别可能的情感分析部分
                overall = []
                positive = []
                neutral = []
                negative = []
                key_statements = []

                # 提取各部分内容
                lines = text.split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    lower_line = line.lower()

                    # 识别部分
                    if "总体" in lower_line or "overall" in lower_line:
                        current_section = "overall"
                        continue
                    elif "正面" in lower_line or "积极" in lower_line or "positive" in lower_line:
                        current_section = "positive"
                        continue
                    elif "中性" in lower_line or "neutral" in lower_line:
                        current_section = "neutral"
                        continue
                    elif "负面" in lower_line or "消极" in lower_line or "negative" in lower_line:
                        current_section = "negative"
                        continue
                    elif "关键" in lower_line or "表述" in lower_line or "key" in lower_line:
                        current_section = "key_statements"
                        continue

                    # 收集要点
                    if current_section == "overall" and (
                            line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        overall.append(line)
                    elif current_section == "positive" and (
                            line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        positive.append(line)
                    elif current_section == "neutral" and (
                            line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        neutral.append(line)
                    elif current_section == "negative" and (
                            line.startswith('-') or line.startswith('•') or line.startswith('*')):
                        line = "- " + line[1:].strip()
                        negative.append(line)
                    elif current_section == "key_statements":
                        # 对于关键表述，检查是否有引号和编号
                        if '"' in line or '"' in line or "'" in line:
                            # 提取引号内的内容
                            start = max(line.find('"'), line.find('"'), line.find("'"))
                            end = max(line.rfind('"'), line.rfind('"'), line.rfind("'"))
                            if start != -1 and end != -1 and start < end:
                                quote = line[start:end + 1]
                                # 格式化为数字加引号
                                key_statements.append(f"{len(key_statements) + 1}. {quote}")
                        elif line.strip() and not line[0].isdigit():
                            # 如果没有引号但有内容，加上引号
                            key_statements.append(f'{len(key_statements) + 1}. "{line.strip()}"')

                # 确保每个部分至少有基本内容
                if not overall:
                    overall = ["- Sentimental score: 0", "- Overall rating: the text is emotionally neutral"]
                elif not any("score" in item for item in overall):
                    # 如果没有评分项，添加一个默认评分
                    overall.insert(0, "- Sentimental score: 0")

                if not positive:
                    positive = ["- No clear positive emotions were identified"]
                if not neutral:
                    neutral = ["- The overall emotional expression of the text is neutral"]
                if not negative:
                    negative = ["- No significant negative feelings identified"]
                if not key_statements:
                    key_statements = ['1. "Failure to extract a clear expression of emotion"']

                # 重组为正确格式
                restructured = (
                        "## Overall sentimental tendencies\n" + "\n".join(overall) + "\n\n" +
                        "## Positive\n" + "\n".join(positive) + "\n\n" +
                        "## Neutral\n" + "\n".join(neutral) + "\n\n" +
                        "## Negative\n" + "\n".join(negative) + "\n\n" +
                        "## Key Emotional Expressions\n" + "\n".join(key_statements)
                )

                return restructured

            # 如果无法修复，返回原文本
            return text

        # 最多重试3次，确保获得正确格式
        max_retries = 3
        current_retry = 0

        while current_retry < max_retries:
            # 检查格式是否正确
            format_correct = check_format(analysis_type, analysis_text)
            bullet_points_correct = check_bullet_points(analysis_text)

            if format_correct and bullet_points_correct:
                break  # 格式正确，不需要重试

            # 生成重试提示
            retry_prompt = generate_retry_prompt(analysis_type, analysis_text)

            # 记录重试次数和原因
            print(
                f"Retry #{current_retry + 1}：{'Formatting error' if not format_correct else ''} {'Missing list items' if not bullet_points_correct else ''}")

            # 降低温度，尝试再次生成
            retry_temperature = max(0.1, 0.3 - current_retry * 0.1)  # 逐次降低温度

            retry_response = ollama.chat(
                model="granite3.2:8b",
                messages=[
                    {"role": "user", "content": final_prompt},
                    {"role": "assistant", "content": analysis_text},
                    {"role": "user", "content": retry_prompt}
                ],
                options={"temperature": retry_temperature}
            )

            if "message" in retry_response:
                new_text = retry_response["message"]["content"].strip()

                # 检查新文本是否比原文本更符合格式要求
                new_format_correct = check_format(analysis_type, new_text)
                new_bullet_points_correct = check_bullet_points(new_text)

                # 只有当新文本格式更好时才替换
                if (new_format_correct and not format_correct) or (
                        new_format_correct == format_correct and new_bullet_points_correct and not bullet_points_correct):
                    analysis_text = new_text
                elif not new_format_correct and not format_correct:
                    # 尝试从原始回答中提取内容并重组
                    restructured_text = attempt_restructure(analysis_type, analysis_text)
                    if check_format(analysis_type, restructured_text):
                        analysis_text = restructured_text

            current_retry += 1

        # 如果所有重试都失败，尝试最后的紧急修复
        if not check_format(analysis_type, analysis_text):
            emergency_text = attempt_restructure(analysis_type, analysis_text)
            if check_format(analysis_type, emergency_text):
                analysis_text = emergency_text

        # 生成PDF报告
        pdf_buffer = io.BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        pdf.setTitle(f"{analysis_type} analysis")

        margin = 50
        line_height = 15
        font_size = 12
        page_width, page_height = letter

        pdf.setFont("Helvetica-Bold", 16)
        title = f"{analysis_type} Analysis Report"
        title_width = pdf.stringWidth(title, "Helvetica-Bold", 16)
        pdf.drawString((page_width - title_width) / 2, page_height - margin - 20, title)

        y_position = page_height - margin - 50
        pdf.setFont("Helvetica", font_size)
        text_object = pdf.beginText(margin, y_position)
        text_object.setLeading(line_height)
        max_line_width = page_width - 2 * margin

        # 改进PDF生成，增加标题和项目符号的差异化显示
        lines = analysis_text.split("\n")
        for paragraph in lines:
            # 识别标题行
            if paragraph.startswith('##'):
                # 如果是标题，先换行，再用粗体显示
                text_object.textLine('')
                pdf.drawText(text_object)

                # 重置文本对象为粗体
                text_object = pdf.beginText(margin, text_object.getY() - line_height)
                text_object.setFont("Helvetica-Bold", font_size + 2)
                text_object.textLine(paragraph.strip())
                pdf.drawText(text_object)

                # 重置文本对象为常规字体
                text_object = pdf.beginText(margin, text_object.getY() - line_height)
                text_object.setFont("Helvetica", font_size)
                text_object.setLeading(line_height)
                continue

            # 识别项目符号行
            elif paragraph.strip().startswith('-'):
                indented_paragraph = '  ' + paragraph  # 添加缩进
                w_list = indented_paragraph.split()
            else:
                w_list = paragraph.split()

            if not w_list:
                text_object.textLine('')
                continue

            line_acc = []
            for w in w_list:
                if pdf.stringWidth(' '.join(line_acc + [w]), "Helvetica", font_size) < max_line_width:
                    line_acc.append(w)
                else:
                    text_object.textLine(' '.join(line_acc))
                    line_acc = [w]
                    if text_object.getY() - line_height < margin:
                        pdf.drawText(text_object)
                        pdf.showPage()
                        pdf.setFont("Helvetica", font_size)
                        y_position = page_height - margin
                        text_object = pdf.beginText(margin, y_position)
                        text_object.setLeading(line_height)

            if line_acc:
                text_object.textLine(' '.join(line_acc))

        pdf.drawText(text_object)
        pdf.save()
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
# 5. Routing: generating word clouds
# ---------------------------

@app.post("/generate_wordcloud")
async def generate_wordcloud_api(request: AnalysisRequest):
    try:
        text = request.pdf_text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text content is empty, can't generate word cloud map")

        wc_buffer = generate_wordcloud_image(text)
        wordcloud_base64 = base64.b64encode(wc_buffer.getvalue()).decode("utf-8")
        return JSONResponse(content={"wordcloud": wordcloud_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# 6. Routing: four-quadrant or histogram visualisation
# ---------------------------

@app.post("/generate_visualization")
async def generate_visualization(request: VisualizationRequest):
    try:
        # 松散处理分析类型以避免大小写、空格等问题
        raw_type = request.analysis_type
        analysis_type = raw_type.lower().strip()
        text = request.analysis_text.strip() if request.analysis_text else ""

        if not text:
            raise HTTPException(status_code=400, detail="The analysis text is empty and no visualisation can be generated")

        if "swot" in analysis_type:
            # 使用改进后的解析器函数
            sections = parse_swot_analysis(text)
            # 使用新函数生成可视化
            fig, ax = generate_swot_visualization(sections)


        elif "pestle" in analysis_type:
            # 使用优化的PESTLE分析可视化
            sections = parse_pestle_analysis(text)
            fig, ax = generate_pestle_visualization(sections)


        elif "most" in analysis_type:
            # Parse MOST content
            sections = parse_most_analysis(text)
            # Use our improved MOST visualization function
            fig, ax = generate_most_visualization(sections)
            # Set appropriate figure size with more space for text
            fig.set_size_inches(14, 10)
            fig.tight_layout(pad=2.0)


        elif "sentiment" in analysis_type:
            # Extract sentiment score from the text analysis
            score = parse_sentiment_analysis(text)
            # Generate the visualization using the score
            fig, ax = generate_sentiment_visualization(score)

        else:
            raise HTTPException(status_code=400, detail="Types of analyses not supported")

        # 输出为图片
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return JSONResponse(content={"chart": chart_base64})
    except Exception as e:
        # 提供详细的错误信息以便调试
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # 在服务器日志中记录详细错误
        raise HTTPException(status_code=500, detail=f"Visualisation generation failure: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # 启动FastAPI应用
    uvicorn.run(app, host="0.0.0.0", port=8000)








