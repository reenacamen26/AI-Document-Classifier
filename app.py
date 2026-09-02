import streamlit as st
import joblib
from pypdf import PdfReader
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from pathlib import Path


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Document Classifier",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "classifier.pkl"
DATASET_PATH = BASE_DIR / "dataset"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load the trained model.")
    st.code(str(e))
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "dataset_results" not in st.session_state:
    st.session_state["dataset_results"] = []

if "uploaded_results" not in st.session_state:
    st.session_state["uploaded_results"] = []
if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = "dataset"


# =========================================================
# TITLE
# =========================================================

st.title("📄 AI Document Classification System")

st.write(
    "Upload, classify, search and analyze your documents using Machine Learning."
)


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text_from_path(file_path):
    """Read TXT or PDF file from the dataset."""

    try:
        if file_path.suffix.lower() == ".txt":
            return file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        elif file_path.suffix.lower() == ".pdf":
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = ""

                for page in reader.pages:
                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

                return text

    except Exception:
        return ""

    return ""


def extract_text_from_uploaded_file(file):
    """Read TXT or PDF uploaded through Streamlit."""

    try:
        # Reset file position before reading
        file.seek(0)

        if file.name.lower().endswith(".pdf"):
            reader = PdfReader(file)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            return text

        elif file.name.lower().endswith(".txt"):
            return file.read().decode(
                "utf-8",
                errors="ignore"
            )

    except Exception:
        return ""

    return ""


# =========================================================
# INVOICE INFORMATION EXTRACTION
# =========================================================

def extract_invoice_information(text):

    information = {}

    patterns = {
        "Invoice Number":
            r"Invoice Number\s*:\s*(.+)",

        "Customer":
            r"Customer\s*:\s*(.+)",

        "Date":
            r"Date\s*:\s*(.+)",

        "Subtotal":
            r"Subtotal\s*:\s*₹?\s*([\d,]+)",

        "GST":
            r"GST\s*:\s*₹?\s*([\d,]+)",

        "Total Amount":
            r"Total Amount\s*:\s*₹?\s*([\d,]+)"
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            value = match.group(1).strip()

            if key in [
                "Subtotal",
                "GST",
                "Total Amount"
            ]:
                value = "₹" + value

            information[key] = value

    return information


# =========================================================
# SUMMARY GENERATION
# =========================================================

def generate_summary(text, max_sentences=3):

    try:
        sentences = sent_tokenize(text)

    except Exception:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return "No summary available."

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    return " ".join(
        sentences[:max_sentences]
    )


# =========================================================
# CLASSIFY TEXT
# =========================================================

def classify_text(text):

    prediction = model.predict(
        [text]
    )[0]

    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            [text]
        )[0]

        confidence = max(
            probabilities
        ) * 100

    return str(prediction), round(confidence, 2)


# =========================================================
# LOAD DATASET DOCUMENTS
# =========================================================

def get_dataset_files():

    files = []

    if not DATASET_PATH.exists():
        return files

    # Expected structure:
    #
    # dataset/
    #   Bank Statement/
    #   Certificate/
    #   Invoice/
    #   Research Paper/
    #   Resume/

    for category_folder in DATASET_PATH.iterdir():

        if category_folder.is_dir():

            for file_path in category_folder.iterdir():

                if file_path.suffix.lower() in [
                    ".txt",
                    ".pdf"
                ]:
                    files.append(file_path)

    return sorted(files)


# =========================================================
# ANALYZE DATASET
# =========================================================

def analyze_dataset():

    results = []

    dataset_files = get_dataset_files()

    for file_path in dataset_files:

        text = extract_text_from_path(
            file_path
        )

        if not text.strip():
            continue

        # ---------------------------------------------
        # ACTUAL CATEGORY FROM DATASET FOLDER
        # ---------------------------------------------

        actual_category = file_path.parent.name

        # ---------------------------------------------
        # AI MODEL PREDICTION
        # ---------------------------------------------

        prediction, confidence = classify_text(
            text
        )

        # ---------------------------------------------
        # SAVE RESULT
        # ---------------------------------------------

        results.append({

            "Document":
                str(
                    file_path.relative_to(
                        DATASET_PATH
                    )
                ),

            "Category":
                prediction,

            "Actual Category":
                actual_category,

            "Confidence":
                confidence,

            "Text":
                text,

            "Source":
                "Dataset"
        })

    return results

# =========================================================
# SAVE UPLOADED RESULT
# =========================================================

def save_uploaded_result(document_name, prediction, confidence, text):
    """Keep every uploaded document as a separate row."""

    new_result = {
        "Document": document_name,
        "Category": prediction,
        "Confidence": confidence,
        "Text": text,
        "Source": "Uploaded"
    }

    # Same filename = update; new filename = add a new row.
    for i, item in enumerate(st.session_state["uploaded_results"]):
        if item["Document"] == document_name:
            st.session_state["uploaded_results"][i] = new_result
            return

    st.session_state["uploaded_results"].append(new_result)


# =========================================================
# DATASET STATUS
# =========================================================

dataset_files = get_dataset_files()

st.info(
    f"📁 Dataset documents available: {len(dataset_files)}"
)


# =========================================================
# UPLOAD ADDITIONAL DOCUMENTS
# =========================================================

st.subheader("📁 Upload Additional Documents")

uploaded_files = st.file_uploader(
    "Choose TXT or PDF documents",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="batch_document_uploader"
)


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    load_dataset_button = st.button(
        "🚀 Analyze All Dataset Documents",
        use_container_width=True
    )

with col2:

    analyze_upload_button = st.button(
        "🔍 Analyze Uploaded Documents",
        use_container_width=True
    )


# =========================================================
# ANALYZE DATASET
# =========================================================

if load_dataset_button:

    with st.spinner(
        "Analyzing all dataset documents..."
    ):

        dataset_results = analyze_dataset()

    st.session_state["dataset_results"] = dataset_results
    st.session_state["analysis_mode"] = "dataset"

    st.success(
        f"✅ Successfully analyzed "
        f"{len(dataset_results)} dataset documents!"
    )


# =========================================================
# ANALYZE ADDITIONAL UPLOADED DOCUMENTS
# =========================================================

if analyze_upload_button:

    if not uploaded_files:

        st.warning(
            "⚠️ Please upload at least one document."
        )

    else:
        st.session_state["analysis_mode"] = "uploaded"

        st.session_state["uploaded_results"] = []

        analyzed_count = 0

        with st.spinner(
            "Analyzing uploaded documents..."
        ):

            for file in uploaded_files:

                text = extract_text_from_uploaded_file(file)

                if not text.strip():
                    continue

                prediction, confidence = classify_text(text)

                # Add this document without deleting older uploads.
                save_uploaded_result(
                    file.name,
                    prediction,
                    confidence,
                    text
                )

                analyzed_count += 1

        st.success(
            f"✅ Successfully analyzed "
            f"{analyzed_count} uploaded documents!"
        )


# =========================================================
# SELECT RESULTS TO DISPLAY
# =========================================================

if st.session_state.get("analysis_mode") == "uploaded":

    # Show only uploaded documents
    all_results = list(
        st.session_state["uploaded_results"]
    )

else:

    # Show dataset documents
    all_results = list(
        st.session_state["dataset_results"]
    )

# =========================================================
# REMOVE DUPLICATES
# =========================================================

unique_results = {}

for result in all_results:

    key = (
        result["Document"],
        result["Source"]
    )

    unique_results[key] = result

results = list(
    unique_results.values()
)

# =========================================================
# SORT DOCUMENTS NATURALLY
# =========================================================

CATEGORY_ORDER = {
    "Bank Statement": 0,
    "Certificate": 1,
    "Invoice": 2,
    "Research Paper": 3,
    "Resume": 4
}


def document_sort_key(result):
    """
    Keep categories in a fixed order and sort document numbers
    numerically: document_1, document_2, ... document_10.
    """

    category = result.get(
        "Actual Category",
        result.get("Category", "")
    )

    category_position = CATEGORY_ORDER.get(
        category,
        999
    )

    document_name = str(
        result.get("Document", "")
    )

    # Handle dataset paths such as:
    # Bank Statement\document_10.txt
    filename = Path(
        document_name.replace("\\", "/")
    ).name

    match = re.search(
        r"document[_-]?(\d+)",
        filename,
        re.IGNORECASE
    )

    if match:
        document_number = int(
            match.group(1)
        )
    else:
        # Files such as new_invoice.txt are placed after
        # numbered dataset documents in their category.
        document_number = 999999

    return (
        category_position,
        document_number,
        document_name.lower()
    )


results = sorted(
    results,
    key=document_sort_key
)


# =========================================================
# DISPLAY RESULTS
# =========================================================


if results:

    # =====================================================
    # CLASSIFICATION RESULTS
    # =====================================================

    st.subheader(
        "📊 Classification Results"
    )

    # =====================================================
    # SEARCH
    # =====================================================

    search = st.text_input(
        "🔎 Search documents",
        placeholder="Type document name...",
        key="document_search"
    )

    # =====================================================
    # CATEGORY FILTER
    # =====================================================

    filter_categories = [
        "All",
        "Bank Statement",
        "Certificate",
        "Invoice",
        "Research Paper",
        "Resume"
    ]

    selected_category = st.selectbox(
        "🗂️ Filter by category",
        filter_categories,
        key="category_filter"
    )

    # =====================================================
    # FILTER RESULTS
    # =====================================================

    filtered_results = results.copy()

    if search:

        filtered_results = [
            r
            for r in filtered_results
            if search.lower()
            in r["Document"].lower()
        ]

    if selected_category != "All":

        filtered_results = [
            r
            for r in filtered_results
            if r["Category"] == selected_category
        ]

    # =====================================================
    # DISPLAY TABLE
    # =====================================================

    display_results = [
        {
            "Document": r["Document"],
            "Category": r["Category"],
            "Confidence": f'{r["Confidence"]:.2f}%',
            "Source": r["Source"]
        }
        for r in filtered_results
    ]

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # AI DOCUMENT DASHBOARD
    # =====================================================

    st.subheader(
        "📊 AI Document Dashboard"
    )

    total_documents = len(results)

    avg_confidence = (
        sum(r["Confidence"] for r in results)
        / total_documents
        if total_documents > 0
        else 0
    )

    highest_confidence = (
        max(r["Confidence"] for r in results)
        if total_documents > 0
        else 0
    )

    category_counts = {
        "Bank Statement": 0,
        "Certificate": 0,
        "Invoice": 0,
        "Research Paper": 0,
        "Resume": 0
    }

    for result in results:

        category = result["Category"]

        if category in category_counts:
            category_counts[category] += 1

    number_of_categories = sum(
        1
        for count in category_counts.values()
        if count > 0
    )

    # =====================================================
    # DASHBOARD METRICS
    # =====================================================

    dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = (
        st.columns(4)
    )

    with dashboard_col1:

        st.metric(
            "📄 Total Documents",
            total_documents
        )

    with dashboard_col2:

        st.metric(
            "🎯 Average Confidence",
            f"{avg_confidence:.2f}%"
        )

    with dashboard_col3:

        st.metric(
            "🏆 Highest Confidence",
            f"{highest_confidence:.2f}%"
        )

    with dashboard_col4:

        st.metric(
            "🗂️ Categories",
            number_of_categories
        )

    # =====================================================
    # CATEGORY DISTRIBUTION
    # =====================================================

    st.subheader(
        "📈 Documents by Category"
    )

    category_df = pd.DataFrame(
        {
            "Category": list(category_counts.keys()),
            "Documents": list(category_counts.values())
        }
    )

    st.bar_chart(
        category_df.set_index("Category")
    )

    # =====================================================
    # DOCUMENT STATISTICS
    # =====================================================

    st.subheader(
        "📈 Document Statistics"
    )

    stats_col1, stats_col2, stats_col3 = st.columns(3)

    with stats_col1:

        st.metric(
            "Total Documents",
            total_documents
        )

    with stats_col2:

        st.metric(
            "Documents Found",
            len(filtered_results)
        )

    with stats_col3:

        st.metric(
            "Average Confidence",
            f"{avg_confidence:.2f}%"
        )

    # =====================================================
    # CATEGORY SUMMARY
    # =====================================================

    st.subheader(
        "📊 Category Summary"
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write(
            f"🏦 **Bank Statements:** "
            f"{category_counts['Bank Statement']}"
        )

        st.write(
            f"📜 **Certificates:** "
            f"{category_counts['Certificate']}"
        )

    with summary_col2:

        st.write(
            f"🧾 **Invoices:** "
            f"{category_counts['Invoice']}"
        )

        st.write(
            f"📑 **Research Papers:** "
            f"{category_counts['Research Paper']}"
        )

    with summary_col3:

        st.write(
            f"📄 **Resumes:** "
            f"{category_counts['Resume']}"
        )

    # =====================================================
    # DETAILED ANALYSIS
    # =====================================================

    st.subheader(
        "📄 Detailed Analysis"
    )

    if not filtered_results:

        st.warning(
            "No documents match your search/filter."
        )

    else:

        for index, result in enumerate(filtered_results):

            document_name = result["Document"]
            prediction = result["Category"]
            confidence = result["Confidence"]
            text = result["Text"]
            source = result["Source"]

            with st.container(border=True):

                st.markdown(
                    f"## 📄 {document_name}"
                )

                st.caption(
                    f"Source: {source}"
                )

                # =============================================
                # CLASSIFICATION
                # =============================================

                st.write(
                    "### 📊 Classification"
                )

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:

                    st.metric(
                        "Category",
                        prediction
                    )

                with detail_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                st.progress(
                    min(max(int(confidence), 0), 100)
                )

                # =============================================
                # INVOICE INFORMATION
                # =============================================

                if prediction == "Invoice":

                    information = extract_invoice_information(text)

                    st.write(
                        "### 🔍 Invoice Information"
                    )

                    if information:

                        invoice_col1, invoice_col2 = st.columns(2)
                        items = list(information.items())

                        for i, (key, value) in enumerate(items):

                            if i % 2 == 0:

                                with invoice_col1:

                                    st.write(
                                        f"**{key}:** {value}"
                                    )

                            else:

                                with invoice_col2:

                                    st.write(
                                        f"**{key}:** {value}"
                                    )

                    else:

                        st.warning(
                            "No invoice information detected."
                        )

                # =============================================
                # DOCUMENT SUMMARY
                # =============================================

                st.write(
                    "### 📝 Document Summary"
                )

                summary = generate_summary(text)

                st.info(summary)

                # =============================================
                # EXTRACTED TEXT
                # =============================================

                st.write(
                    "### 📄 Extracted Text"
                )

                st.text_area(
                    "Document Content",
                    text[:5000],
                    height=200,
                    key=f"text_{source}_{index}_{document_name}"
                )

else:

    st.warning(
        "No documents analyzed yet."
    )

    st.write(
        "Click **🚀 Analyze All Dataset Documents** "
        "to classify your dataset documents."
    )


# ============================================================
# UPLOAD & CLASSIFY NEW DOCUMENT
# ============================================================

st.divider()

st.subheader(
    "📤 Upload & Classify New Document"
)

st.write(
    "Upload a single TXT or PDF document and classify it "
    "using the trained machine-learning model."
)

single_file = st.file_uploader(
    "Choose a TXT or PDF document",
    type=["txt", "pdf"],
    accept_multiple_files=False,
    key="single_document_uploader"
)

if single_file is not None:

    classify_new_button = st.button(
        "🤖 Classify New Document",
        use_container_width=True,
        key="classify_new_document_button"
    )

    if classify_new_button:

        with st.spinner(
            "Classifying new document..."
        ):

            new_text = extract_text_from_uploaded_file(
                single_file
            )

            if not new_text.strip():

                st.error(
                    "❌ Could not extract text from the document."
                )

            else:

                new_prediction, new_confidence = classify_text(
                    new_text
                )

                # Update the same filename instead of creating
                # another duplicate uploaded result.
                save_uploaded_result(
                    single_file.name,
                    new_prediction,
                    new_confidence,
                    new_text
                )

                st.success(
                    f"✅ Document classified as "
                    f"**{new_prediction}** "
                    f"with **{new_confidence:.2f}%** confidence."
                )

                st.write(
                    f"**Document:** {single_file.name}"
                )

                st.write(
                    f"**Category:** {new_prediction}"
                )

                st.write(
                    f"**Confidence:** {new_confidence:.2f}%"
                )

                st.write(
                    "### 📝 Summary"
                )

                st.info(
                    generate_summary(new_text)
                )

                st.write(
                    "### 📄 Extracted Text"
                )

                st.text_area(
                    "Document Content",
                    new_text[:5000],
                    height=200,
                    key="new_document_extracted_text"
                )


# ============================================================
# END OF APPLICATION
# ============================================================