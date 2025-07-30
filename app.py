# app.py

import os
import json
import pickle
import base64
from io import BytesIO
import time
import threading
import hashlib

from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify
from openai import OpenAI
import pandas as pd
from pdf2image import convert_from_path
from dotenv import load_dotenv
from PIL import Image
import cv2
import numpy as np
from openpyxl import Workbook
# --- START OF THE FIX for Poppler Path ---
poppler_bin_path = r"C:\poppler-24.08.0\Library\bin"
if poppler_bin_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] += os.pathsep + poppler_bin_path
    print(f"DEBUG: Successfully added Poppler to PATH: {poppler_bin_path}")
# --- END OF THE FIX ---

# Load environment variables from .env file
load_dotenv()

# --- Typhoon API Setup ---
TYPHOON_API_KEY = os.getenv('TYPHOON_API_KEY')

if not TYPHOON_API_KEY:
    raise ValueError("TYPHOON_API_KEY must be set in the .env file.")

# Initialize separate clients for clarity
ocr_client = OpenAI(base_url="https://api.opentyphoon.ai/v1", api_key=TYPHOON_API_KEY)
instruct_client = OpenAI(base_url="https://api.opentyphoon.ai/v1", api_key=TYPHOON_API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a-default-secret-key')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Rate Limiter and Caching Classes (Unchanged) ---
class RateLimiter:
    def __init__(self, requests_per_minute):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.lock = threading.Lock()
        self.last_request_time = 0
    def wait(self):
        if self.interval == 0: return
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.interval:
                sleep_time = self.interval - time_since_last
                print(f"DEBUG: Rate limiting. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            self.last_request_time = time.time()

class SimpleCache:
    def __init__(self, cache_file="typhoon_cache.pkl"):
        self.cache_file = cache_file; self.cache = {}; self._load_cache()
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f: self.cache = pickle.load(f)
    def _save_cache(self):
        with open(self.cache_file, "wb") as f: pickle.dump(self.cache, f)
    def get_response(self, client_instance, model, messages, **kwargs):
        key_dict = {"model": model, "messages": messages, **kwargs}
        key = hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            print(f"DEBUG: Using cached response for key {key}"); return self.cache[key]
        print(f"DEBUG: No cache found for key {key}. Making new API call.")
        response = client_instance.chat.completions.create(model=model, messages=messages, **kwargs)
        self.cache[key] = response; self._save_cache(); return response

CACHE_FILE = os.path.join(UPLOAD_FOLDER, "typhoon_cache.pkl")
api_cache = SimpleCache(cache_file=CACHE_FILE)
ocr_rate_limiter = RateLimiter(requests_per_minute=20)
instruct_rate_limiter = RateLimiter(requests_per_minute=200)

# --- Image Pre-processing Function (Unchanged) ---
def enhance_image_from_base64(img_base64: str) -> str:
    try:
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        _, buffer = cv2.imencode('.png', thresh)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"DEBUG: Error in image enhancement: {e}")
        return img_base64

# --- FULLY REDESIGNED AND CORRECTED FUNCTION: extract_table_from_image ---
def extract_table_from_image(img_base64):
    # --- Stage 1: High-Accuracy Transcription ---
    print("DEBUG: Stage 1 - Enhancing and transcribing image...")
    enhanced_base64 = enhance_image_from_base64(img_base64)
    transcription_prompt = "You are an OCR engine. Transcribe the image and return the result in a JSON object with a single key `natural_text`."
    messages_stage1 = [{"role": "user", "content": [{"type": "text", "text": transcription_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{enhanced_base64}"}}]}]
    
    try:
        ocr_rate_limiter.wait()
        response_stage1 = api_cache.get_response(ocr_client, model="typhoon-ocr-preview", messages=messages_stage1, max_tokens=4096, temperature=0.0)
        raw_output = response_stage1.choices[0].message.content
        transcribed_text = json.loads(raw_output.strip().strip("```json"))['natural_text']
        print("DEBUG: Stage 1 - Transcription successful.")
    except Exception as e:
        print(f"DEBUG: Error in Stage 1 (Transcription): {e}")
        raise ValueError(f"Failed to transcribe image with Typhoon-OCR: {e}")

    # --- Stage 2: Structuring the Data (With Corrected Logic) ---
    print("DEBUG: Stage 2 - Structuring transcribed text...")
    
    # --- THIS IS THE NEW, BULLETPROOF PROMPT ---
    structuring_system_prompt = """
    You are an automated data extraction engine. Your task is to analyze the provided markdown text from a Thai construction quote and convert it into a structured JSON object.

    Your response MUST strictly adhere to the following JSON schema:
    {
      "description": "string",
      "table_items": [
        {
          "order": "integer",
          "name": "string",
          "quantity": "number",
          "unit": "string",
          "material_unit_price": "number",
          "material_total": "number",
          "labor_unit_price": "number",
          "labor_total": "number",
          "grand_total": "number"
        }
      ]
    }

    Rules:
    - Your final response MUST be ONLY the valid JSON object and nothing else.
    - Do not include markdown wrappers like ```json.
    - Do not add any explanations, apologies, or conversational text.
    - If a field's value is not found in the text, use a sensible default (0 for numbers, "" for strings, [] for the table_items array).
    - Convert all prices and quantities to numbers (integer or float), removing currency symbols or commas.
    """
    
    structuring_user_prompt = f"Here is the transcribed text. Convert it to the specified JSON schema:\n---\n{transcribed_text}\n---"
    messages_stage2 = [{"role": "system", "content": structuring_system_prompt}, {"role": "user", "content": structuring_user_prompt}]

    try:
        instruct_rate_limiter.wait()
        response_stage2 = api_cache.get_response(
            instruct_client,
            model="typhoon-v2.1-12b-instruct",
            messages=messages_stage2,
            temperature=0.0,
            max_tokens=4096,
        )
        
        # --- NEW: More Robust JSON Cleaning ---
        # This will find and extract the JSON even if the model wraps it in text.
        json_str = response_stage2.choices[0].message.content
        
        start_index = json_str.find('{')
        end_index = json_str.rfind('}')
        
        if start_index != -1 and end_index != -1:
            json_str_cleaned = json_str[start_index : end_index + 1]
            print("DEBUG: Stage 2 - Structuring successful.")
            return json.loads(json_str_cleaned)
        else:
            # This will be triggered if the model fails to return any JSON at all
            raise ValueError("Model did not return a valid JSON object in its response.")

    except Exception as e:
        print(f"DEBUG: Error in Stage 2 (Structuring): {e}")
        raw_response_content = "Could not retrieve response content."
        if 'response_stage2' in locals() and response_stage2.choices:
            raw_response_content = response_stage2.choices[0].message.content
        raise ValueError(f"Failed to structure data. Raw model output: '{raw_response_content}'")

# --- Original Flask Routes (Unchanged) ---
def cleanup_old_files():
    import time
    current_time = time.time(); one_hour = 3600
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path) and current_time - os.path.getmtime(file_path) > one_hour:
                os.remove(file_path)
    except Exception as e: print(f"DEBUG: Error cleaning up old files: {e}")

cleanup_old_files()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files: flash('No file part'); return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '': flash('No selected file'); return redirect(request.url)
        if file and file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            return redirect(url_for('extract', filename=file.filename))
        else:
            flash('Please upload a PDF file.'); return redirect(request.url)
    return render_template('index.html')

@app.route('/extract/<filename>', methods=['GET'])
def extract(filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        images = convert_from_path(file_path)
        pages_data = []
        for page_num, image in enumerate(images, 1):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            try:
                extracted_data = extract_table_from_image(img_base64)
            except Exception as e:
                flash(f'Error extracting data from page {page_num}: {str(e)}')
                extracted_data = {"description": f"Error on page {page_num}", "table_items": []}
            pages_data.append({'page_number': page_num, 'image_base64': img_base64, 'extracted_data': extracted_data})
        session_file = file_path.replace('.pdf', '_session.pkl')
        with open(session_file, 'wb') as f: pickle.dump(pages_data, f)
        return redirect(url_for('review', filename=filename))
    except Exception as e:
        import traceback
        error_msg = f'Error in extract function: {str(e)}\n{traceback.format_exc()}'
        print(f"DEBUG: {error_msg}")
        flash(error_msg); return redirect(url_for('index'))

@app.route('/review/<filename>')
def review(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    session_file = file_path.replace('.pdf', '_session.pkl')
    with open(session_file, 'rb') as f:
        pages_data = pickle.load(f)
    return render_template('review.html', filename=filename, pages_data=pages_data)

@app.route('/update_data', methods=['POST'])
def update_data():
    data = request.json
    filename = data['filename']; page_number = data['page_number']; updated_items = data['items']
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    session_file = file_path.replace('.pdf', '_session.pkl')
    with open(session_file, 'rb') as f: pages_data = pickle.load(f)
    for page_data in pages_data:
        if page_data['page_number'] == page_number:
            page_data['extracted_data']['table_items'] = updated_items
            break
    with open(session_file, 'wb') as f: pickle.dump(pages_data, f)
    return jsonify({'status': 'success'})

@app.route('/validate/<filename>')
def validate(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    session_file = file_path.replace('.pdf', '_session.pkl')
    with open(session_file, 'rb') as f: pages_data = pickle.load(f)
    combined_items = []; all_descriptions = []
    for page_data in pages_data:
        extracted_data = page_data['extracted_data']
        description = extracted_data.get("description", "")
        items = extracted_data.get("table_items", [])
        if description: all_descriptions.append(f"Page {page_data['page_number']}: {description}")
        for item in items:
            item_with_page = item.copy()
            item_with_page['source_page'] = page_data['page_number']
            combined_items.append(item_with_page)
    for i, item in enumerate(combined_items, 1): item['order'] = i
    combined_data = {
        'description': ' | '.join(all_descriptions),
        'table_items': combined_items,
        'total_pages': len(pages_data),
        'total_items': len(combined_items)
    }
    combined_file = file_path.replace('.pdf', '_combined.pkl')
    with open(combined_file, 'wb') as f: pickle.dump(combined_data, f)
    return render_template('validate.html', filename=filename, combined_data=combined_data)

def cleanup_files(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    base_name = file_path.replace('.pdf', '')
    files_to_remove = [file_path, f"{base_name}_session.pkl", f"{base_name}_combined.pkl"]
    page_num = 1
    while True:
        page_image = f"{base_name}_page_{page_num}.jpg"
        if os.path.exists(page_image): files_to_remove.append(page_image); page_num += 1
        else: break
    for file_to_remove in files_to_remove:
        try:
            if os.path.exists(file_to_remove): os.remove(file_to_remove)
        except Exception as e: print(f"DEBUG: Failed to remove {file_to_remove}: {e}")

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    combined_file = file_path.replace('.pdf', '_combined.pkl')
    try:
        with open(combined_file, 'rb') as f: combined_data = pickle.load(f)
    except FileNotFoundError:
        flash('Please validate your data first.'); return redirect(url_for('review', filename=filename))
    output = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = 'Combined Data'
    # Add combined description as header
    description = combined_data.get('description', '')
    if description:
        ws.append([description])
        ws.append([])  # Empty row for spacing
    
    # Add summary info
    ws.append([f"Total Pages: {combined_data.get('total_pages', 0)}"])
    ws.append([f"Total Items: {combined_data.get('total_items', 0)}"])
    ws.append([])  # Empty row for spacing
    
    # Add column headers
    items = combined_data.get('table_items', [])
    if items:
        ws.append(list(items[0].keys()))
        
        # Add data rows
        for item in items:
            ws.append(list(item.values()))
    else:
        ws.append(["No table data found in the PDF"])
    
    wb.save(output)
    output.seek(0)
    
    # Clean up files after creating Excel
    cleanup_files(filename)
    
    return send_file(output, download_name=filename.replace('.pdf', '_combined.xlsx'), as_attachment=True)


def test():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, 'นนน,พรสุทธิ-Arbor_จิกน้ำ.pdf')
        if os.path.exists(file_path):
            images = convert_from_path(file_path)
            return f'Success: {len(images)} pages converted. Type: {type(images)}'
        else:
            return 'Test PDF not found'
    except Exception as e:
        import traceback
        return f'Error: {str(e)}<br><pre>{traceback.format_exc()}</pre>'

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 