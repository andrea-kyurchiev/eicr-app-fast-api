"""
EICR Processor
=========================

A standalone tool for extracting data from EICR PDFs without a GUI.
Returns data as a Python dictionary/JSON structure.

Usage:
    from eicr_processor import EICRProcessor
    
    processor = EICRProcessor()
    data = processor.process_pdf("path/to/report.pdf")
    print(data)
"""

import sys
import os
import re
import pandas as pd

# --- Dependency Checks ---
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Required library 'PyMuPDF' not found. Please install it.")

try:
    import camelot
except ImportError:
    raise ImportError("Required library 'camelot-py' not found. Please install it.")

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    print("Warning: 'doctr' library not found. Metadata extraction (DB Name, Location, etc.) will be skipped.")

# --- Helper Functions for Table Extraction ---

def _is_string_cell(x):
    if pd.isna(x):
        return False
    s = str(x).strip()
    return any(c.isalpha() for c in s)

def _make_unique(cols):
    out = []
    seen = {}
    for c in cols:
        c_clean = c if c != "" else "col"
        cnt = seen.get(c_clean, 0)
        seen[c_clean] = cnt + 1
        out.append(f"{c_clean}" + (f"_{cnt}" if cnt else ""))
    return out
def _truncate_footer_rows(df):
    """
    Detects if a row contains keywords indicating the start of the 
    'Engineer and Test Instruments' footer and cuts the dataframe there.
    """
    # Keywords found in the blue header or the form below the table
    # stop_markers = [
    #     "ENGINEER AND TEST INSTRUMENTS",
    #     "TEST INSTRUMENTS FOR DB",
    #     "MFT", 
    #     "TESTED BY",
    #     "CONTINUITY"
    # ]
    
    cutoff_idx = None
    
    for idx, row in df.iterrows():
        # Join all cells in the row to a single string for searching
        # use .dropna() to ignore NaNs and .upper() for case-insensitive match
        row_text = " ".join(row.dropna().astype(str)).upper()
        
        # Check if the row contains the footer header
        # if any(marker in row_text for marker in stop_markers):
        if "ENGINEER AND TEST" in row_text:
            cutoff_idx = idx
            break
            
    if cutoff_idx is not None:
        # Return everything UP TO (but not including) the footer row
        return df.iloc[:cutoff_idx].copy()
    
    return df
    
def _remove_trailing_empty_rows(df):
    def is_empty_row(row):
        return all((pd.isna(x) or str(x).strip() == "") for x in row)
    last_valid_idx = None
    for i in reversed(df.index):
        if not is_empty_row(df.loc[i]):
            last_valid_idx = i
            break
    if last_valid_idx is None:
        return df.iloc[0:0]
    return df.loc[:last_valid_idx].reset_index(drop=True)

def _clean_camelot_df(raw_df):
    header_row_idx = None
    for i, row in raw_df.iterrows():
        if all(_is_string_cell(val) for val in row):
            header_row_idx = i
            break
    if header_row_idx is None:
        # Fallback or strict error? 
        # For headless processing, we might want to return empty DF or raise error.
        # Keeping it consistent with previous logic:
        raise ValueError("No suitable header row found.")

    header = raw_df.iloc[header_row_idx].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().tolist()
    body = raw_df.iloc[header_row_idx + 1 :].copy().reset_index(drop=True)
    header = _make_unique(header)
    body.columns = header
    df = _truncate_footer_rows(body)
    final_df = _remove_trailing_empty_rows(df)
    return final_df

def extract_table(pdf_path, page=1, flavor='lattice', table_index=0, suppress_stdout=True):
    """
    Reads a table from `pdf_path` at `page` using Camelot and returns a cleaned DataFrame.
    """
    tables = camelot.read_pdf(
        pdf_path,
        pages=str(page),
        flavor=flavor,
        suppress_stdout=suppress_stdout,
        split_text=True,
        line_scale=30
    )
    if len(tables) == 0:
        raise ValueError(f"No tables found on page {page}")

    if table_index >= len(tables) or table_index < 0:
        raise IndexError(f"table_index {table_index} out of range")

    raw_df = tables[table_index].df
    cleaned_df = _clean_camelot_df(raw_df)
    return cleaned_df

# --- Main Processor Class ---

class EICRProcessor:
    def __init__(self):
        self.ocr_model = None

    def _get_ocr_model(self):
        """Lazy loader for OCR model"""
        if DOCTR_AVAILABLE and self.ocr_model is None:
            # print("Loading OCR model (this may take a moment)...")
            self.ocr_model = ocr_predictor(pretrained=True)
        return self.ocr_model

    def _extract_circuit_data(self, pdf_path, cd_page, tr_page):
        if tr_page > cd_page:
            pages_to_process = range(cd_page, tr_page)
        else:
            pages_to_process = [cd_page]

        dfs = []
        for page in pages_to_process:
            try:
                part_df = extract_table(pdf_path, page)
                dfs.append(part_df)
            except ValueError:
                continue
        
        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)

    def _extract_test_data(self, pdf_path, cd_page, tr_page):
        if cd_page is not None and tr_page > cd_page:
            page_count = tr_page - cd_page
        else:
            page_count = 1

        if page_count > 1:
            pages_to_process = range(tr_page, tr_page + page_count)
        else:
            pages_to_process = [tr_page]
        
        dfs = []
        for page in pages_to_process:
            try:
                part_df = extract_table(pdf_path, page)
                dfs.append(part_df)
            except ValueError:
                continue
        
        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)

    def _extract_metadata_with_doctr(self, pdf_path, page_num):
        if not DOCTR_AVAILABLE:
            return {}
        
        model = self._get_ocr_model()
        if model is None:
            return {}

        try:
            doc = DocumentFile.from_pdf(pdf_path)
            
            # Adjust 1-based page_num to 0-based index
            page_idx = page_num - 1
            if page_idx < 0 or page_idx >= len(doc):
                return {}

            result = model([doc[page_idx]])
            
            lines = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        text = " ".join(w.value for w in line.words).strip()
                        if text:
                            lines.append(text)
            
            meta = {
                "DB name": None,
                "Location": None,
                "No of phases": None,
                "Supplied from": None
            }

            for i, item in enumerate(lines):
                text = item.strip()

                if text.startswith("DB name"):
                    meta["DB name"] = text.replace("DB name", "").strip()

                if text.startswith("Location"):
                    meta["Location"] = text.replace("Location", "").strip()

                if "Phase sequence confirmed" in text and i - 1 >= 0:
                    before = lines[i - 1]
                    m = re.search(r"(\d+)", before)
                    if m:
                        meta["No of phases"] = m.group(1)
                
                if text == "Supply polarity confirmed" and i - 1 >= 0:
                    meta["Supplied from"] = lines[i - 1].strip()
            
            return {k: v for k, v in meta.items() if v is not None}

        except Exception as e:
            print(f"OCR Error on page {page_num}: {e}")
            return {}

    def process_pdf(self, pdf_path):
        """
        Parses the PDF and returns the extracted EICR data as a dictionary.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # print(f"Processing {pdf_path}...")
        
        # 1. Parse PDF Structure using PyMuPDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"Could not open PDF with PyMuPDF: {e}")

        circuit_details_str = "CIRCUIT DETAILS"
        test_results_str = "TEST RESULTS"
        
        pages_with_circuit_details = []
        pages_with_test_results = []
        board_names_found = []

        # Pass 1: Identify pages
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text")
            if circuit_details_str in text:
                pages_with_circuit_details.append(i + 1)
            if test_results_str in text:
                pages_with_test_results.append(i + 1)

        # Pass 2: Extract Board Names via Regex
        for i in pages_with_test_results:
            text = doc.load_page(i - 1).get_text("text")
            found_board = "Unknown Board"
            for line in text.splitlines():
                if 'TEST RESULTS' in line:
                    m = re.search(r'(?i)test results(.*)', line)
                    if m:
                        candidate = m.group(1).strip()
                        if candidate:
                            found_board = candidate
                            break
            board_names_found.append(found_board)
        total_boards = len(board_names_found)
        print(f"Total boards found: {total_boards}")
        doc.close()

        # Align Data
        min_len = min(len(board_names_found), len(pages_with_circuit_details), len(pages_with_test_results))
        
        # 2. Extract Data for each board
        all_boards_data = []

        if min_len == 0:
            print("No complete board sections found.")
            return {"Boards": []}

        for idx in range(min_len):
            board_name = board_names_found[idx]
            cd_page = pages_with_circuit_details[idx]
            tr_page = pages_with_test_results[idx]
            
            # print(f"  > Extracting Board: {board_name} (CD Page: {cd_page}, TR Page: {tr_page})")

            # Extract Tables
            try:
                c_df = self._extract_circuit_data(pdf_path, cd_page, tr_page)
                c_data = c_df.to_dict(orient='records') if c_df is not None else []
            except Exception as e:
                print(f"    Error extracting circuit tables: {e}")
                c_data = []

            try:
                t_df = self._extract_test_data(pdf_path, cd_page, tr_page)
                t_data = t_df.to_dict(orient='records') if t_df is not None else []
            except Exception as e:
                print(f"    Error extracting test tables: {e}")
                t_data = []

            # Extract Metadata (OCR)
            meta = self._extract_metadata_with_doctr(pdf_path, cd_page)
            
            # Construct Board Object
            board_obj = {
                "DB name": meta.get("DB name") or board_name,
                "Location": meta.get("Location") or "Unknown",
                "No of phases": meta.get("No of phases") or "Unknown",
                "Supplied from": meta.get("Supplied from") or "Unknown",
                "Circuit Details": c_data,
                "Test Results": t_data
            }
            all_boards_data.append(board_obj)

        final_output = {
            "Boards": all_boards_data
        }
        
        return final_output
