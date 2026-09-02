import os
import pandas as pd


# ==========================================
# DATASET FOLDER
# ==========================================

DATASET_PATH = "dataset"


# ==========================================
# READ DOCUMENTS
# ==========================================

documents = []


# Go through every category folder
for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(
        DATASET_PATH,
        category
    )

    # Ignore files such as documents.csv
    if not os.path.isdir(category_path):
        continue

    # Read every .txt file
    for filename in os.listdir(category_path):

        if filename.lower().endswith(".txt"):

            file_path = os.path.join(
                category_path,
                filename
            )

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    text = file.read()

                if text.strip():

                    documents.append({
                        "Text": text,
                        "Category": category
                    })

            except Exception as e:

                print(
                    f"Could not read {file_path}: {e}"
                )


# ==========================================
# CREATE DATAFRAME
# ==========================================

data = pd.DataFrame(documents)


# ==========================================
# SAVE CSV
# ==========================================

data.to_csv(
    "dataset/documents.csv",
    index=False,
    encoding="utf-8"
)


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("\n==========================================")
print("DATASET CREATED SUCCESSFULLY!")
print("==========================================")

print(
    f"Total documents: {len(data)}"
)

print("\nCategories:")

print(
    data["Category"].value_counts()
)

print("\nCSV columns:")

print(
    list(data.columns)
)

print(
    "\nFile: dataset/documents.csv"
)