import tabula
import pandas as pd
import PyPDF2
import pdfplumber
import io


def get_text_after_marker(pdf_path, page_number):
    """Extract text after marker using pdfplumber. The marker should not be included in the text."""
    marker = (
        "Transactions - Numéro de carte 4569 59XX XXXX 4657 - Jean-Baptiste Poullet"
    )

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        text = page.extract_text()

        if marker in text:
            print("Found marker in page!")
            print("Original text length:", len(text))
            start_idx = text.find(marker)
            cleaned_text = text[start_idx:]
            print("Cleaned text length:", len(cleaned_text))

            # Get the y position of the marker
            words = page.extract_words()
            for word in words:
                if marker.split()[0] in word["text"]:  # Look for first word of marker
                    marker_y = word["top"]
                    return True, marker_y
    return False, None


pdf_path = "7196329135_22_02_2025_16_47_06.pdf"


def debug_print_pdf_content(buffer):
    """Print the content of a PDF buffer for debugging"""
    buffer.seek(0)
    reader = PyPDF2.PdfReader(buffer)
    print("\n=== PDF Content ===")
    for i, page in enumerate(reader.pages):
        print(f"\n--- Page {i + 1} ---")
        text = page.extract_text()
        print(text[:500] + "..." if len(text) > 500 else text)
    print("=== End PDF Content ===\n")
    buffer.seek(0)


# Create a new PDF with cleaned pages
with open(pdf_path, "rb") as file:
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_writer = PyPDF2.PdfWriter()

    # Process all pages except the last one
    found_marker = False
    for page_num in range(len(pdf_reader.pages) - 1):
        page = pdf_reader.pages[page_num]
        has_marker, marker_y = get_text_after_marker(pdf_path, page_num)

        if has_marker:
            found_marker = True
            # Add the page directly (no copy needed)
            pdf_writer.add_page(page)
            # Get the page from the writer to modify it
            new_page = pdf_writer.pages[-1]

            # Use different crop offsets for first page vs others
            crop_offset = -110 if page_num == 0 else -70

            # Adjust both mediabox and cropbox to crop the top portion
            crop_y = float(new_page.mediabox[3]) - marker_y + crop_offset

            # Update both mediabox and cropbox
            new_page.mediabox.top = crop_y
            if hasattr(new_page, "/CropBox"):
                new_page.cropbox.top = crop_y

            print(
                f"Added cropped page {page_num} with crop at y={crop_y} (offset={crop_offset})"
            )

    if not found_marker:
        print("Card transaction marker not found in the PDF!")
        exit(1)

    # Save to a temporary buffer
    temp_buffer = io.BytesIO()
    pdf_writer.write(temp_buffer)

    # Debug print the content
    debug_print_pdf_content(temp_buffer)

    # Save the temp_buffer to a file for debugging
    with open("debug_cleaned.pdf", "wb") as debug_file:
        temp_buffer.seek(0)
        debug_file.write(temp_buffer.getvalue())

    temp_buffer.seek(0)


def merge_split_rows(df):
    """Merge rows that are actually continuations of previous rows"""
    merged_rows = []
    current_row = None

    for idx, row in df.iterrows():
        # Check if this is a continuation row (first two columns are NaN)
        if pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]):
            if current_row is not None:
                # Find the first non-NaN value and merge it with the corresponding column
                for col_idx, val in enumerate(row):
                    if pd.notna(val):
                        current_val = current_row.iloc[col_idx]
                        current_row.iloc[col_idx] = f"{current_val} {val}".strip()
        else:
            if current_row is not None:
                merged_rows.append(current_row)
            current_row = row.copy()

    if current_row is not None:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows, columns=df.columns)


def clean_dataframe(df):
    """Clean the dataframe but preserve all columns"""
    # Only drop columns that are completely empty (all NaN)
    df = df.dropna(axis=1, how="all")
    return df


# Now process the cleaned PDF with tabula
dfs = tabula.read_pdf(
    temp_buffer,
    pages="all",
    multiple_tables=True,
    lattice=False,
    guess=True,
    pandas_options={"header": None},
)

# Process each table and merge split rows
processed_tables = []
for df in dfs:
    if not df.empty:
        # First clean the dataframe by removing problematic columns
        df = clean_dataframe(df)
        # Then merge split rows
        df = merge_split_rows(df)
        processed_tables.append(df)
dfs = processed_tables

# Debug print
print("\nRaw tables found (after merging split rows):")
for i, df in enumerate(dfs):
    print(f"\nTable {i + 1}:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())


def clean_header(df):
    """Clean and format table columns based on content patterns"""
    print("\nProcessing table content:")
    print(df.head())

    # Create a copy of the dataframe
    clean_df = df.copy()

    # Define expected columns in order they appear in PDF
    expected_pdf_columns = [
        "DATE TRANSACTION",
        "DATE COMPTABILISATION",
        "DESCRIPTION",
        "FOURNISSEUR",
        "PAYS",
        "DEVISE",  # Optional column
        "MONTANT",
    ]

    # Debug print column structure
    print(f"\nFound {len(df.columns)} columns:")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: Sample values: {df[col].head().tolist()}")

    # Assign column names based on actual column count
    if len(df.columns) == 6:
        # Standard format without DEVISE
        columns = expected_pdf_columns[:5] + [expected_pdf_columns[-1]]
    elif len(df.columns) == 7:
        # Format with DEVISE column
        columns = expected_pdf_columns
    else:
        print(f"Unexpected number of columns: {len(df.columns)}")
        return None

    clean_df.columns = columns

    # Clean up the amount column
    def parse_amount(row):
        """Parse amount considering possible DEVISE column"""
        # Get amount from the last column
        amount_str = str(row.iloc[-1])

        # If DEVISE exists and has USD conversion, use that amount
        if len(row) > 6 and pd.notna(row["DEVISE"]):
            devise_info = str(row["DEVISE"])
            if "USD" in devise_info:
                try:
                    # Extract USD amount (format: "10,00 USD 1 EUR = 1,02249488USD")
                    usd_amount = float(
                        devise_info.split("USD")[0].replace(",", ".").strip()
                    )
                    # Extract EUR conversion rate
                    conversion_rate = float(
                        devise_info.split("EUR =")[1]
                        .replace("USD", "")
                        .replace(",", ".")
                        .strip()
                    )
                    # Calculate EUR amount and round to 2 decimal places
                    amount = round(usd_amount / conversion_rate, 2)
                    return -amount  # These are usually payments, so negative
                except (ValueError, IndexError) as e:
                    print(
                        f"Warning: Could not parse USD conversion '{devise_info}': {e}"
                    )

        # For non-USD amounts, process normally
        amount_str = (
            amount_str.replace("EUR", "")
            .replace("€", "")
            .replace("\u202f", "")
            .replace("\xa0", "")
            .strip()
        )
        is_negative = "-" in amount_str
        amount_str = amount_str.replace("-", "").strip()
        amount_str = amount_str.replace(",", ".")
        try:
            amount = round(float(amount_str), 2)
            return -amount if is_negative else amount
        except ValueError:
            print(f"Warning: Could not parse amount '{amount_str}'")
            return 0.0

    # Apply the parsing function to each row
    clean_df["MONTANT"] = clean_df.apply(parse_amount, axis=1)

    # Remove DEVISE column drop - we want to keep it
    return clean_df


# Update expected columns to include DEVISE
expected_columns = [
    "DATE TRANSACTION",
    "DATE COMPTABILISATION",
    "DESCRIPTION",
    "FOURNISSEUR",
    "CODE PAYS",
    "DEVISE",  # Added DEVISE column
    "MONTANT",
]

# Process each table found
processed_dfs = []
for i, df in enumerate(dfs):
    print(f"\nProcessing table {i + 1}")
    cleaned_df = clean_header(df)
    if cleaned_df is not None and not cleaned_df.empty:
        # Add empty columns required by the format
        cleaned_df["FOURNISSEUR"] = ""
        cleaned_df["CODE PAYS"] = cleaned_df["PAYS"]  # Use PAYS as CODE PAYS

        # Reorder columns to match expected format
        cleaned_df = cleaned_df.reindex(columns=expected_columns)
        processed_dfs.append(cleaned_df)

if not processed_dfs:
    print("No tables found with the required columns!")
    exit(1)

# Combine all processed tables
final_df = pd.concat(processed_dfs, ignore_index=True)

# Save to CSV
final_df.to_csv("transactions.csv", index=False)

# Display the result
print("\nFinal DataFrame:")
print(final_df.head())
