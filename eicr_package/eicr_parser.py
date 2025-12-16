# eicr_parser.py
from typing import Optional, Dict, Any
import re
from doctr.io import DocumentFile
# make sure `model` is a doctr OCR predictor you pass in from the calling code

rx_report = re.compile(r"REPORT No:\s*(EICR-[\dA-Za-z\-]+)")

def get_confidence_for_text(target_text: str, page) -> Optional[float]:
    """Return confidence for the first word of target_text on a doctr Page."""
    first_word = target_text.split()[0]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                if word.value.strip() == first_word:
                    return round(word.confidence, 2)
    return None

def get_confidence_for_line(target_line: str, page) -> Optional[float]:
    """Return average confidence of the words in the line (matching exact text)."""
    target_line = target_line.strip().lower()
    for block in page.blocks:
        for line in block.lines:
            line_text = " ".join(w.value for w in line.words).strip().lower()
            if line_text == target_line and line.words:
                return round(sum(w.confidence for w in line.words) / len(line.words), 2)
    return None

def _default_output_keys():
    keys = [
        "Report Number","Client Name","Client Address","Client Town","Client County",
        "Client Postcode","Report Reason","Inspection Date","Installation Occupier",
        "Installation Address","Installation Town","Installation County",
        "Installation Postcode","Extent of Installation","Overall Condition","Created at"
    ]
    return {k: {"value": None, "confidence": None} for k in keys}

def get_eicr_info(pdf_file: str, model, pages: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
    """
    Parse EICR PDF and return a dict of fields: {"Field": {"value": ..., "confidence": ...}, ...}

    Args:
      pdf_file: path-like to PDF
      model: doctr OCR predictor (pretrained or your loaded model). Required.
      pages: optional list of zero-based page indices to analyze (default first two pages)
    """
    # load doc
    doc = DocumentFile.from_pdf(pdf_file)

    # default pages: first two pages
    pages = pages if pages is not None else [0, 1]
    # create tiny sub-documents to feed into model (doctr model expects Document-like objects)
    p0 = [doc[pages[0]]]
    p1 = [doc[pages[1]]]

    result0 = model(p0)
    result1 = model(p1)

    page0 = result0.pages[0]
    page1 = result1.pages[0]

    # extract lines
    def page_lines(page):
        lines = []
        for block in page.blocks:
            for line in block.lines:
                text = " ".join(w.value for w in line.words).strip()
                if text:
                    lines.append(text)
        return lines

    lines0 = page_lines(page0)
    lines1 = page_lines(page1)

    out = _default_output_keys()

    # Page 0 parsing (report number, overall condition, installation block, issued on)
    for i, line in enumerate(lines0):
        m = rx_report.search(line)
        if m:
            out["Report Number"]["value"] = m.group(1)
            out["Report Number"]["confidence"] = get_confidence_for_text(out["Report Number"]["value"], page0)

        if 'And was deemed to be:'.lower() in line.lower():
            cond_idx = i + 1
            if cond_idx < len(lines0):
                out["Overall Condition"]["value"] = lines0[cond_idx]
                out["Overall Condition"]["confidence"] = get_confidence_for_line(lines0[cond_idx], page0)

        if 'BS7671:2018+A3:2024' in line:
            installation_idx = i + 1
            # guard indexes
            for j, key in enumerate(["Installation Address","Installation Town","Installation County","Installation Postcode"]):
                idx = installation_idx + j
                if idx < len(lines0):
                    out[key]["value"] = lines0[idx]
                    out[key]["confidence"] = get_confidence_for_line(lines0[idx], page0)

        if 'Issued on' in line:
            issue_idx = i + 1
            if issue_idx < len(lines0):
                out["Created at"]["value"] = lines0[issue_idx]
                out["Created at"]["confidence"] = get_confidence_for_line(lines0[issue_idx], page0)

    # Page 1 parsing: locate section start indices (safe defaults if not found)
    client_start_idx = reason_start_idx = installation_start_idx = extent_start_idx = declaration_start_idx = None
    for i, line in enumerate(lines1):
        up = line.upper()
        if "DETAILS OF THE CLIENT" in up:
            client_start_idx = i
        if "REASONS FOR PRODUCING THIS REPORT" in up:
            reason_start_idx = i
        if "DETAILS OF THE INSTALLATION" in up:
            installation_start_idx = i
        if "EXTENT AND LIMITATIONS" in up:
            extent_start_idx = i
        if "DECLARATION" in up:
            declaration_start_idx = i

    # helper to safely check ranges
    def safe_range(a, b, default_b):
        if a is None or b is None:
            return None, None
        return a, b

    # client details
    if client_start_idx is not None and reason_start_idx is not None:
        for i in range(client_start_idx, reason_start_idx):
            l = lines1[i]
            if "Client name" in l:
                idx = i + 2
                if idx < len(lines1):
                    out["Client Name"]["value"] = lines1[idx]
                    out["Client Name"]["confidence"] = get_confidence_for_line(lines1[idx], page1)
            if "Address" in l:
                idx = i + 2
                if idx < len(lines1):
                    out["Client Address"]["value"] = lines1[idx]
                    out["Client Address"]["confidence"] = get_confidence_for_line(lines1[idx], page1)
            if "Town" in l:
                idx = i + 2
                if idx < len(lines1):
                    out["Client Town"]["value"] = lines1[idx]
                    out["Client Town"]["confidence"] = get_confidence_for_line(lines1[idx], page1)
            if "County" in l:
                idx = i + 2
                if idx < len(lines1):
                    out["Client County"]["value"] = lines1[idx]
                    out["Client County"]["confidence"] = get_confidence_for_line(lines1[idx], page1)
            if "Postcode" in l:
                idx = i + 4
                if idx < len(lines1):
                    out["Client Postcode"]["value"] = lines1[idx]
                    out["Client Postcode"]["confidence"] = get_confidence_for_line(lines1[idx], page1)

    # reason details
    if reason_start_idx is not None and installation_start_idx is not None:
        for i in range(reason_start_idx, installation_start_idx):
            l = lines1[i]
            if "Reasons for producing this report" in l:
                idx = i + 2
                if idx < len(lines1):
                    out["Report Reason"]["value"] = lines1[idx]
                    out["Report Reason"]["confidence"] = get_confidence_for_line(lines1[idx], page1)
            if "Date inspection carried out" in l:
                idx = i + 2
                if idx < len(lines1):
                    out["Inspection Date"]["value"] = lines1[idx]
                    out["Inspection Date"]["confidence"] = get_confidence_for_line(lines1[idx], page1)

    # installation details
    if installation_start_idx is not None and extent_start_idx is not None:
        for i in range(installation_start_idx, extent_start_idx):
            if "Occupier name" in lines1[i]:
                idx = i + 4
                if idx < len(lines1):
                    out["Installation Occupier"]["value"] = lines1[idx]
                    out["Installation Occupier"]["confidence"] = get_confidence_for_line(lines1[idx], page1)

    if extent_start_idx is not None and extent_start_idx + 2 < len(lines1):
        out["Extent of Installation"]["value"] = lines1[extent_start_idx + 2]
        out["Extent of Installation"]["confidence"] = get_confidence_for_line(lines1[extent_start_idx + 2], page1)

    return out

# optional: define __all__ for clarity
__all__ = ["get_eicr_info"]
