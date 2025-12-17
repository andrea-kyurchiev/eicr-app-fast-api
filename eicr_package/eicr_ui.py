import ipywidgets as widgets
import tempfile
import json
import os
import time
from IPython.display import display
from doctr.models import ocr_predictor
import fitz
from eicr_package.eicr_parser import get_eicr_info
from eicr_package.extractor import EICRSupplyExtractor
from eicr_package.eicr_boards import EICRProcessor

class EICRProcessorUI:
    def __init__(self, template_path="./eicr_package/template.png"):
        self.template_path = template_path
        self.output_data = None
        self.temp_pdf_path = None
        self.output_name = None
        # Initialize UI components
        self._init_widgets()
        
    def _init_widgets(self):
        # 1. Widgets
        self.upload = widgets.FileUpload(
            accept=".pdf",
            multiple=False,
            description="Upload PDF"
        )

        self.process_btn = widgets.Button(
            description="Process PDF",
            button_style="primary",
            icon="cogs"
        )

        self.save_btn = widgets.Button(
            description="Save JSON",
            button_style="success",
            disabled=True,
            icon="download"
        )

        self.status = widgets.HTML("<b>Status:</b> Waiting for PDF upload")

        # 2. Event Binding
        self.process_btn.on_click(self._on_process_clicked)
        self.save_btn.on_click(self._on_save_clicked)
        self.upload.observe(self._on_upload_change, names='value')
        
        # 3. Layout
        self.layout = widgets.VBox([
            widgets.HTML("<h3>EICR PDF Processor</h3>"),
            self.upload,
            self.process_btn,
            self.save_btn,
            self.status
        ])
    
    def _on_upload_change(self, change):
        """Callback to update status when a file is uploaded."""
        if change['new']:
            self.status.value = "<b>Status:</b> File uploaded"
    
    def _get_supply_char_page_no(self, pdf_path):
        doc = fitz.open(pdf_path)
        s1 = "DETAILS OF THE COMPANY"
        s2 = "SUPPLY CHARACTERISTICS AND EARTHING ARRANGEMENTS"
        s3 = "PARTICULARS OF INSTALLATION"
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text")
            if s1 in text and s2 in text and s3 in text:
                target_page_index = i
                break
        return target_page_index
        
    def process_eicr_pdf(self, pdf_path):
        """Internal logic to process the PDF using the package."""
        start_time = time.time()
        # Load model (cached if possible, but here we load per call as per your snippet)
        ocr_model = ocr_predictor(pretrained=True)

        eicr_main_record = get_eicr_info(pdf_path, ocr_model)
        self.output_name = eicr_main_record["Report Number"]["value"]

        page_num = self._get_supply_char_page_no(pdf_path)
        extractor = EICRSupplyExtractor(template_path=self.template_path)
        supply_characteristics, particulars_of_installation = extractor.extract(
            pdf_path,
            page_number=page_num
        )

        processor = EICRProcessor()
        boards_data = processor.process_pdf(pdf_path)

        merged = {
            "eicr_main_record": eicr_main_record,
            "supply_characteristics": supply_characteristics,
            "particulars_of_installation": particulars_of_installation,
            **boards_data
        }
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Print to console (visible in the notebook output log)
        print(f"Total Extraction Time: {elapsed:.2f} seconds")
        
        return merged

    def _on_process_clicked(self, b):
        if not self.upload.value:
            self.status.value = "<b>Status:</b> No PDF uploaded"
            return

        self.status.value = "<b>Status:</b> Processing... (Please wait)"
        self.process_btn.disabled = True

        try:
            # Handle ipywidgets structure (v7 vs v8 compatibility)
            # v8 returns tuple, v7 returns dict
            uploaded_vals = self.upload.value
            if isinstance(uploaded_vals, tuple):
                uploaded_file = uploaded_vals[0]
            else:
                uploaded_file = next(iter(uploaded_vals.values()))
            
            # Access content
            pdf_bytes = uploaded_file['content'] if 'content' in uploaded_file else uploaded_file.content

            # Save uploaded pdf temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(pdf_bytes)
                self.temp_pdf_path = f.name

            # Run processing
            self.output_data = self.process_eicr_pdf(self.temp_pdf_path)
            
            # Update UI
            self.save_btn.disabled = False
            self.status.value = "<b>Status:</b> Processing completed"

        except Exception as e:
            self.status.value = f"<b>Status:</b> Error: {str(e)}"
            # Print full trace to console for debugging
            import traceback
            traceback.print_exc()
        finally:
            self.process_btn.disabled = False

    def _on_save_clicked(self, b):
        if self.output_data is None:
            self.status.value = "<b>Status:</b> Nothing to save"
            return
        
        # --- Generate Dynamic Output Name ---
        if self.output_name:
            base_name = self.output_name
            output_path = f"{base_name}_output.json"
        else:
            output_path = "final_output.json"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.output_data, f, indent=2, ensure_ascii=False)
            
            # Check for Google Colab to trigger browser download
            try:
                from google.colab import files
                files.download(output_path)
                self.status.value = f"<b>Status:</b> ⬇Downloading {output_path}..."
            except ImportError:
                self.status.value = f"<b>Status:</b> Saved locally as '{output_path}'"
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> Save Error: {str(e)}"

    def show(self):
        """Display the UI widget."""
        display(self.layout)
