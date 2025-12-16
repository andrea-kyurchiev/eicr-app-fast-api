import cv2
import numpy as np
from pathlib import Path
from boxdetect import config
from boxdetect.pipelines import get_boxes
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from .utils import (
    pdf_page_to_cv2_image, 
    find_missing_ref_index, 
    get_int_before_token, 
    get_float_before_token, 
    get_num_after_token
)

class EICRSupplyExtractor:
    def __init__(self, template_path="template.png"):
        self.template_path = template_path
        self.ocr_model = ocr_predictor(pretrained=True)
        
    def _get_boxes_from_crop(self, crop_img):
        """crop_img: numpy array"""

        # Configure BoxDetect
        cfg = config.PipelinesConfig()
        cfg.width_range = (10,20)
        cfg.height_range = (10,20)
        cfg.scaling_factors = [0.7, 1.0, 1.3]
        cfg.wh_ratio_range = (0.8, 1.2)
        cfg.group_size_range = (1, 2)
        cfg.dilation_iterations = 0

        rects, grouping_rects, _, _ = get_boxes(crop_img, cfg=cfg, plot=False)
        return grouping_rects

    def process_earthing(self, img):
        crop_x = 60
        crop_y = 394
        crop_h = 353
        crop_w = 140
        crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        grouping_rects = self._get_boxes_from_crop(crop)
        
        # Logic: The MISSING box is the selected one (assuming filled boxes aren't detected)
        ref = [98, 145, 190, 235, 281]
        detected_x = [b[1] for b in grouping_rects] # boxdetect returns (x, y, w, h) ? check user logic

        missing_val = None
        for r in ref:
            if not any(abs(r - v) <= 4 for v in detected_x):
                missing_val = r
                break
        
        mapping = {98: 'TN-S', 145: 'TN-C-S', 190: 'TN-C', 235: 'TT', 281: 'IT'}
        return mapping.get(missing_val)

    def process_supply_type(self, img):
        crop_x = 205
        crop_y = 394
        crop_h = 353
        crop_w = 300
        crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        grouping_rects = self._get_boxes_from_crop(crop)
        
        results = {}
        
        # 1. Current Type (AC/DC) - First rect logic
        if grouping_rects:
            val = grouping_rects[0][0] # User logic used index 0 here.
            # Logic: check distance to ref
            ref_ac_dc = {76: 'a.c.', 260: 'd.c.'}
            found_key = None
            for r in ref_ac_dc.keys():
                if not abs(r - val) <= 4: # User logic: "missing" is the selected one
                     found_key = r
                     break
            results['Current Type'] = ref_ac_dc.get(found_key)
            
            # Remove first element for next step
            grouping_rects = grouping_rects[1:]
        
        # 2. Conductor Type
        conductor_list = [
            '1-phase (2 wire)', '1-phase (3 wire)', '2-phase (3 wire)', 
            '3-phase (3 wire)', '3-phase (4 wire)', '2 pole', '3 pole', 'Other'
        ]

        ref_grid = [
            (76, 149), (175, 149), (76,194), (76, 240), 
            (175, 240), (260, 149), (260, 194), (260, 240)
        ]
        
        missing_indices = find_missing_ref_index(ref_grid, grouping_rects)
        if missing_indices:
            results['Conductor Type'] = conductor_list[missing_indices[0]]
            
        return results

    def process_ocr_data(self, img):
        crop_x = 535
        crop_y = 394
        crop_h = 353
        crop_w = 390
        crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        result = self.ocr_model([crop])
        
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(w.value for w in line.words).strip()
                    if text: lines.append(text)
        
        data = {}

        try:
            v_idx = lines.index('V')
            if v_idx > 0: data['Nominal Voltage'] = int(lines[v_idx - 1])
        except (ValueError, IndexError):
            data['Nominal Voltage'] = None
        for i, line in enumerate(lines):
          if 'Uo' in line:
            uo = lines[i+1]
            data['Uo Voltage'] = int(uo) if uo.isdigit() else None
        
        data['Frequency'] = get_int_before_token(lines, 'HZ')
        data['PFC'] = get_float_before_token(lines, 'kA')
        data['Earth Loop Impedance'] = get_num_after_token(lines, 'Earth loop', float)
        data['Number of Supplies'] = get_num_after_token(lines, 'No of', int)
        
        return data 

    def process_polarity(self, img):
        crop_x = 535
        crop_y = 394
        crop_h = 353
        crop_w = 390
        crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        if not Path(self.template_path).exists():
            return None

        template = cv2.imread(self.template_path, cv2.IMREAD_COLOR)
        if template is None: return None
            
        result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return 'Yes' if max_val >= 0.7 else 'No'
    
    def process_poi(self, img):
        crop_x = 60
        crop_y = 770
        crop_h = 200
        crop_w = 140
        crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        if not Path(self.template_path).exists():
            return None

        template = cv2.imread(self.template_path, cv2.IMREAD_COLOR)
        if template is None: return None
            
        result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        return "Distributor's facility" if max_loc[1] < 100 else "Earth electrode"

    def extract(self, pdf_path, page_number):
        # Convert PDF to Image
        full_img = pdf_page_to_cv2_image(pdf_path, page_number)
        
        characteristics = {
            "Earthing Arrangement": None,
            "Current Type": None,
            "Conductor Type": None,
            "Nominal Voltage": None,
            "Uo Voltage": None,
            "Frequency": None,
            "PFC": None,
            "Earth Loop Impedance": None,
            "Number of Supplies": None,
            "Supply Polarity Confirmed": None
        }
        particulars_of_installation = {
          'Means of Earthing': None
        }

        # Run Extractions
        characteristics['Earthing Arrangement'] = self.process_earthing(full_img)
        
        type_res = self.process_supply_type(full_img)
        characteristics.update(type_res)
        
        ocr_res = self.process_ocr_data(full_img)
        characteristics.update(ocr_res)
        
        characteristics['Supply Polarity Confirmed'] = self.process_polarity(full_img)

        particulars_of_installation['Means of Earthing'] = self.process_poi(full_img)

        return characteristics, particulars_of_installation