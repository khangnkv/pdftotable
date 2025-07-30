# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask web application that extracts table data from PDF files using OpenAI's GPT-4 vision model and converts the extracted data to Excel format. The application specifically targets Thai quote/invoice tables with predefined schema for construction/material quotes.

## Architecture

- **Flask Web App**: Single-file Flask application (`app.py`) serving a simple upload interface
- **PDF Processing Pipeline**: 
  1. PDF upload via web form
  2. PDF-to-image conversion using `pdf2image` 
  3. Base64 image encoding for OpenAI API
  4. GPT-4.1 analysis with structured JSON schema
  5. Excel file generation with extracted data
- **Template**: Simple HTML form (`templates/index.html`) for PDF upload
- **File Storage**: Local `uploads/` directory for temporary file storage

## Key Dependencies

The application relies on these Python packages (not managed by requirements file):
- `flask` - Web framework
- `openai` - OpenAI API client
- `pdfplumber` - PDF text extraction (imported but not actively used)
- `pandas` - Data manipulation for Excel export
- `pdf2image` - PDF to image conversion
- `openpyxl` - Excel file creation
- `base64` - Image encoding for API

## Development Commands

Since this is a simple Flask app without package management files:

**Run the application:**
```bash
python app.py
```

**Install dependencies manually:**
```bash
pip install flask openai pdfplumber pandas pdf2image openpyxl
```

## Configuration

- OpenAI API key is set in `app.py:14` (hardcoded - security concern)
- Flask secret key is hardcoded as 'supersecretkey'
- Upload folder is set to 'uploads/'
- GPT model used: "gpt-4.1" (newly published model with vision capabilities)

## Data Schema

The application expects Thai construction quotes with this JSON structure:
- `description`: String description of the quote
- `items`: Array of line items with:
  - `order`: Integer order number
  - `name`: String item name
  - `quantity`: Number quantity
  - `unit`: String unit of measurement
  - `material_unit_price`: Number material unit price
  - `material_total`: Number material total
  - `labor_unit_price`: Number labor unit price
  - `labor_total`: Number labor total
  - `grand_total`: Number grand total

## File Structure

```
pdf_to_excel_gpt/
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Upload form template
└── uploads/            # Temporary file storage
    ├── testing.py      # Test script for pdf2image
    └── [uploaded files]
```

## Security Considerations

- API key is hardcoded in source code
- No file validation beyond PDF extension check
- Files are stored permanently in uploads/ directory
- No authentication or rate limiting