import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

def pdf_page_to_cv2_image(pdf_path, page_number, zoom=2.0):
    """Render PDF page to cv2 BGR image without saving to disk."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_number]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert directly from bytes to numpy array
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    arr = np.array(img)
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return img_bgr

def find_missing_ref_index(ref, inp, tol=4):
    """Find which reference coordinate is NOT present in the input list."""
    matched = [False] * len(ref)
    for i, (rx, ry) in enumerate(ref):
        for (x, y, _, _) in inp:
            if abs(x - rx) <= tol and abs(y - ry) <= tol:
                matched[i] = True
                break
    
    # Return index of missing items
    missing_indices = [i for i, m in enumerate(matched) if not m]
    return missing_indices

# Text parsing helpers
def get_int_before_token(lst, token):
    indices = [i for i, x in enumerate(lst) if token.lower() in x.lower()]
    if not indices: return None
    idx = indices[0]
    if idx == 0: return None
    try: return int(lst[idx - 1])
    except ValueError: return None

def get_float_before_token(lst, token):
    indices = [i for i, x in enumerate(lst) if token.lower() in x.lower()]
    if not indices: return None
    idx = indices[0]
    if idx == 0: return None
    try: return float(lst[idx - 1])
    except ValueError: return None

def get_num_after_token(lst, token, dtype=int):
    indices = [i for i, x in enumerate(lst) if token.lower() in x.lower()]
    if not indices: return None
    idx = indices[0]
    if idx == len(lst) - 1: return None
    try: return dtype(lst[idx + 1])
    except ValueError: return None