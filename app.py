import os
import json
import pickle
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify
import openai
import pdfplumber
import pandas as pd
from io import BytesIO
from openai import OpenAI
import base64
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GPT-4.1 API setup
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
client = OpenAI(api_key=api_key)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Clean up old files on startup
def cleanup_old_files():
    """Clean up files older than 1 hour"""
    import time
    current_time = time.time()
    one_hour = 3600  # 1 hour in seconds
    
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > one_hour:
                    os.remove(file_path)
                    print(f"DEBUG: Removed old file: {filename}")
    except Exception as e:
        print(f"DEBUG: Error cleaning up old files: {e}")

# Clean up old files on startup
cleanup_old_files()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            return redirect(url_for('extract', filename=file.filename))
        else:
            flash('Please upload a PDF file.')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/extract/<filename>', methods=['GET'])
def extract(filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Convert all pages of PDF to JPG images
        try:
            images = convert_from_path(file_path)
            print(f"DEBUG: Successfully converted PDF, got {len(images)} pages, type: {type(images)}")
        except Exception as e:
            flash(f'Error converting PDF: {str(e)}')
            return redirect(url_for('index'))
        
        # Process each page and store results
        pages_data = []
        
        for page_num, image in enumerate(images, 1):
            print(f"DEBUG: Processing page {page_num}, image type: {type(image)}")
            image_path = file_path.replace('.pdf', f'_page_{page_num}.jpg')
            image.save(image_path, 'JPEG')
            
            # Read and encode image as base64
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Extract data from this page using GPT-4.1
            try:
                extracted_data = extract_table_from_image(img_base64)
            except Exception as e:
                flash(f'Error extracting data from page {page_num}: {str(e)}')
                extracted_data = {"description": "", "table_items": []}
            
            pages_data.append({
                'page_number': page_num,
                'image_path': image_path,
                'image_base64': img_base64,
                'extracted_data': extracted_data
            })
        
        print(f"DEBUG: Successfully processed {len(pages_data)} pages")
        
        # Store data in session or temporary file for review
        session_file = file_path.replace('.pdf', '_session.pkl')
        with open(session_file, 'wb') as f:
            pickle.dump(pages_data, f)
        
        print(f"DEBUG: Successfully saved session file")
        return redirect(url_for('review', filename=filename))
        
    except Exception as e:
        import traceback
        error_msg = f'Error in extract function: {str(e)}\n{traceback.format_exc()}'
        print(f"DEBUG: {error_msg}")
        flash(error_msg)
        return redirect(url_for('index'))

def extract_table_from_image(img_base64):
    """Extract table data using GPT-4.1 vision API"""
    # Build messages for OpenAI API (image input)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the table from the given image and return it only as a JSON object that matches the schema in response_format, Ignore all other text (headers, logos, footers). If any field is missing, leave it blank or as an empty string/zero."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                }
            ]
        }
    ]
    
    # Call OpenAI API with function calling (JSON schema)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "thai_quote_table",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "table_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "order": {"type": "integer"},
                                    "name": {"type": "string"},
                                    "quantity": {"type": "number", "minimum": 0},
                                    "unit": {"type": "string"},
                                    "material_unit_price": {"type": "number", "minimum": 0},
                                    "material_total": {"type": "number", "minimum": 0},
                                    "labor_unit_price": {"type": "number", "minimum": 0},
                                    "labor_total": {"type": "number", "minimum": 0},
                                    "grand_total": {"type": "number", "minimum": 0}
                                },
                                "required": [
                                    "order", "name", "quantity", "unit",
                                    "material_unit_price", "material_total",
                                    "labor_unit_price", "labor_total", "grand_total"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["description", "table_items"],
                    "additionalProperties": False
                }
            }
        },
        temperature=0,
        max_completion_tokens=32000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Get the JSON string from the response
    json_str = response.choices[0].message.content
    return json.loads(json_str)

def cleanup_files(filename):
    """Clean up all files related to the uploaded PDF"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    base_name = file_path.replace('.pdf', '')
    
    files_to_remove = [
        file_path,  # Original PDF
        f"{base_name}_session.pkl",  # Session data
        f"{base_name}_combined.pkl",  # Combined data
    ]
    
    # Add page image files
    page_num = 1
    while True:
        page_image = f"{base_name}_page_{page_num}.jpg"
        if os.path.exists(page_image):
            files_to_remove.append(page_image)
            page_num += 1
        else:
            break
    
    # Remove all files
    for file_to_remove in files_to_remove:
        try:
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
                print(f"DEBUG: Removed file: {file_to_remove}")
        except Exception as e:
            print(f"DEBUG: Failed to remove {file_to_remove}: {e}")
    
    print(f"DEBUG: Cleanup completed for {filename}")

@app.route('/review/<filename>')
def review(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    session_file = file_path.replace('.pdf', '_session.pkl')
    
    # Load extracted data
    with open(session_file, 'rb') as f:
        pages_data = pickle.load(f)
    
    return render_template('review.html', filename=filename, pages_data=pages_data)

@app.route('/update_data', methods=['POST'])
def update_data():
    data = request.json
    filename = data['filename']
    page_number = data['page_number']
    updated_items = data['items']
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    session_file = file_path.replace('.pdf', '_session.pkl')
    
    # Load current data
    with open(session_file, 'rb') as f:
        pages_data = pickle.load(f)
    
    # Update the specific page data
    for page_data in pages_data:
        if page_data['page_number'] == page_number:
            page_data['extracted_data']['table_items'] = updated_items
            break
    
    # Save updated data
    with open(session_file, 'wb') as f:
        pickle.dump(pages_data, f)
    
    return jsonify({'status': 'success'})

@app.route('/validate/<filename>')
def validate(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    session_file = file_path.replace('.pdf', '_session.pkl')
    
    # Load current data
    with open(session_file, 'rb') as f:
        pages_data = pickle.load(f)
    
    # Combine all pages into single table
    combined_items = []
    all_descriptions = []
    
    for page_data in pages_data:
        extracted_data = page_data['extracted_data']
        description = extracted_data.get("description", "")
        items = extracted_data.get("table_items", [])
        
        if description:
            all_descriptions.append(f"Page {page_data['page_number']}: {description}")
        
        # Add page information to each item and append to combined list
        for item in items:
            item_with_page = item.copy()
            item_with_page['source_page'] = page_data['page_number']
            combined_items.append(item_with_page)
    
    # Renumber the order field to be sequential across all pages
    for i, item in enumerate(combined_items, 1):
        item['order'] = i
    
    # Create combined data structure
    combined_data = {
        'description': ' | '.join(all_descriptions),
        'table_items': combined_items,
        'total_pages': len(pages_data),
        'total_items': len(combined_items)
    }
    
    # Save combined data for final download
    combined_file = file_path.replace('.pdf', '_combined.pkl')
    with open(combined_file, 'wb') as f:
        pickle.dump(combined_data, f)
    
    return render_template('validate.html', filename=filename, combined_data=combined_data)

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    combined_file = file_path.replace('.pdf', '_combined.pkl')
    
    # Load combined data
    try:
        with open(combined_file, 'rb') as f:
            combined_data = pickle.load(f)
    except FileNotFoundError:
        flash('Please validate your data first before downloading.')
        return redirect(url_for('review', filename=filename))
    
    # Create Excel file with single sheet containing all data
    from openpyxl import Workbook
    output = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Combined Data"
    
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

@app.route('/test')
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