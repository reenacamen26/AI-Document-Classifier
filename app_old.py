import streamlit as st
import joblib
from pypdf import PdfReader
import re
import joblib
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


model = load_model()


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
    """
    Read TXT or PDF file from dataset.
    """

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

    except Exception as e:

        return ""


def extract_text_from_uploaded_file(file):

    """
    Read TXT or PDF uploaded through Streamlit.
    """

    try:

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

    probabilities = model.predict_proba(
        [text]
    )[0]

    confidence = max(
        probabilities
    ) * 100

    return prediction, round(confidence, 2)


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


def analyze_dataset():

    results = []

    dataset_files = get_dataset_files()

    for file_path in dataset_files:

        text = extract_text_from_path(
            file_path
        )

        if not text.strip():

            continue

        prediction, confidence = classify_text(
            text
        )

        results.append({

            "Document":
                file_path.name,

            "Category":
                prediction,

            "Confidence":
                confidence,

            "Text":
                text,

            "Source":
                "Dataset"

        })

    return results


# =========================================================
# DATASET STATUS
# =========================================================

dataset_files = get_dataset_files()

st.info(
    f"📁 Dataset documents available: {len(dataset_files)}"
)


# =========================================================
# UPLOAD DOCUMENTS
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["txt", "pdf"],
    key="new_document_uploader"
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

    st.success(
        f"✅ Successfully analyzed {len(dataset_results)} dataset documents!"
    )


# =========================================================
# ANALYZE UPLOADED DOCUMENTS
# =========================================================

if analyze_upload_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )

    else:

        upload_results = []

        with st.spinner(
            "Analyzing uploaded documents..."
        ):

            for file in uploaded_files:

                text = extract_text_from_uploaded_file(
                    file
                )

                if not text.strip():

                    continue

                prediction, confidence = classify_text(
                    text
                )

                upload_results.append({

                    "Document":
                        file.name,

                    "Category":
                        prediction,

                    "Confidence":
                        confidence,

                    "Text":
                        text,

                    "Source":
                        "Uploaded"

                })

        st.session_state["upload_results"] = upload_results

        st.success(
            f"✅ Successfully analyzed {len(upload_results)} uploaded documents!"
        )


# =========================================================
# COMBINE RESULTS
# =========================================================

all_results = []

if "dataset_results" in st.session_state:

    all_results.extend(
        st.session_state["dataset_results"]
    )

if "upload_results" in st.session_state:

    all_results.extend(
        st.session_state["upload_results"]
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

        placeholder="Type document name..."
    )


    # =====================================================
    # CATEGORY FILTER
    # =====================================================

    all_categories = [

        "All",

        "Bank Statement",

        "Certificate",

        "Invoice",

        "Research Paper",

        "Resume"
    ]


    selected_category = st.selectbox(

        "🗂️ Filter by category",

        all_categories
    )


    # =====================================================
    # FILTER RESULTS
    # =====================================================

    filtered_results = results.copy()


    # Search
    if search:

        filtered_results = [

            r

            for r in filtered_results

            if search.lower()
            in r["Document"].lower()
        ]


    # Category
    if selected_category != "All":

        filtered_results = [

            r

            for r in filtered_results

            if r["Category"]
            == selected_category
        ]


    # =====================================================
    # DISPLAY TABLE
    # =====================================================

    display_results = [

        {

            "Document":
                r["Document"],

            "Category":
                r["Category"],

            "Confidence":
                f'{r["Confidence"]:.2f}%',

            "Source":
                r["Source"]

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


    # Average confidence

    avg_confidence = (

        sum(
            r["Confidence"]
            for r in results
        )

        / len(results)
    )


    dashboard_col1, dashboard_col2 = st.columns(2)


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


    # =====================================================
    # DOCUMENTS BY CATEGORY
    # =====================================================

    st.subheader(
        "📂 Documents by Category"
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


    st.bar_chart(
        category_counts
    )


    # =====================================================
    # DOCUMENT STATISTICS
    # =====================================================

    st.subheader(
        "📈 Document Statistics"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Total Documents",

            len(results)
        )


    with col2:

        st.metric(

            "Documents Found",

            len(filtered_results)
        )


    with col3:

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


    for index, result in enumerate(
        filtered_results
    ):

        document_name = result["Document"]

        prediction = result["Category"]

        confidence = result["Confidence"]

        text = result["Text"]

        source = result["Source"]


        # =================================================
        # DOCUMENT CARD
        # =================================================

        with st.container(
            border=True
        ):

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


            # Confidence bar

            st.progress(

                min(
                    int(confidence),
                    100
                )
            )


            # =============================================
            # INVOICE INFORMATION
            # =============================================

            if prediction == "Invoice":

                information = (
                    extract_invoice_information(
                        text
                    )
                )


                st.write(
                    "### 🔍 Invoice Information"
                )


                if information:

                    invoice_col1, invoice_col2 = (
                        st.columns(2)
                    )


                    items = list(
                        information.items()
                    )


                    for i, (key, value) in enumerate(
                        items
                    ):

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


            summary = generate_summary(
                text
            )


            st.info(
                summary
            )


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

    # =====================================================
    # EMPTY STATE
    # =====================================================

    st.warning(
        "No documents analyzed yet."
    )

    st.write(
        "Click **🚀 Analyze All Dataset Documents** "
        "to classify your 50 dataset documents."
    )

# ============================================================
# UPLOAD AND CLASSIFY NEW DOCUMENT
# ============================================================

st.markdown("---")

st.header("📤 Upload & Classify New Document")

st.write(
    "Upload a TXT document and let the AI automatically "
    "predict its document category."
)

uploaded_file = st.file_uploader(
    "Choose a TXT document",
    type=["txt"],
    key="new_document_uploader"
)

# Store uploaded classification results
if "uploaded_results" not in st.session_state:
    st.session_state.uploaded_results = []


if uploaded_file is not None:

    # Read uploaded file
    text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    st.subheader("📄 Uploaded Document")
    st.write(f"**File:** {uploaded_file.name}")

    if not text.strip():

        st.warning("⚠️ The uploaded document is empty.")

    else:

        # Show extracted text
        with st.expander("👁️ View Extracted Text"):

            st.text_area(
                "Document Content",
                text,
                height=200,
                disabled=True,
                key="uploaded_text_preview"
            )

        # Classify button
        if st.button(
            "🤖 Classify Document",
            type="primary",
            key="classify_uploaded_document"
        ):

            try:

                # Load trained model
                uploaded_model = joblib.load(
                    "model/classifier.pkl"
                )

                # Predict category
                prediction = uploaded_model.predict(
                    [text]
                )[0]

                # Calculate confidence
                confidence = 0.0

                if hasattr(
                    uploaded_model,
                    "predict_proba"
                ):

                    probabilities = (
                        uploaded_model
                        .predict_proba([text])[0]
                    )

                    confidence = (
                        max(probabilities) * 100
                    )

                # Create result
                new_result = {
                    "Document": uploaded_file.name,
                    "Category": str(prediction),
                    "Confidence": round(
                        confidence,
                        2
                    ),
                    "Source": "Uploaded"
                }

                # Avoid duplicate entries
                existing_names = [
                    item["Document"]
                    for item in st.session_state.uploaded_results
                ]

                if uploaded_file.name not in existing_names:

                    st.session_state.uploaded_results.append(
                        new_result
                    )

                # Success message
                st.success(
                    "✅ Document classified successfully!"
                )

                # Result section
                st.subheader(
                    "📊 Classification Result"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Category",
                        str(prediction)
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                # Summary
                st.subheader(
                    "📝 Document Summary"
                )

                st.info(
                    f"**{uploaded_file.name}** "
                    f"has been classified as "
                    f"**{prediction}** with a confidence "
                    f"of **{confidence:.2f}%**."
                )

                # Extracted text
                st.subheader(
                    "📃 Extracted Text"
                )

                st.text_area(
                    "Document Content",
                    text,
                    height=250,
                    disabled=True,
                    key="uploaded_text_result"
                )

            except Exception as e:

                st.error(
                    "❌ Error while classifying "
                    "the document."
                )

                st.code(str(e))


# ============================================================
# UPLOADED DOCUMENT RESULTS
# ============================================================

if st.session_state.uploaded_results:

    st.markdown("---")

    st.subheader(
        "📋 Uploaded Document Results"
    )

    uploaded_df = pd.DataFrame(
        st.session_state.uploaded_results
    )

    uploaded_df["Confidence"] = (
        uploaded_df["Confidence"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        uploaded_df,
        use_container_width=True,
        hide_index=True
    )