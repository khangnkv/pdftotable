# PDF to Excel Converter

A Flask web application that extracts table data from PDF files using TYPHOON's models and converts the extracted data to Excel format. Specifically designed for Thai construction/material quotes and invoices.

## Features

- Upload PDF files through a web interface
- Convert PDF pages to images for AI processing
- Extract structured table data using Typhoon2.1-Gemma3-12B API
- Review and edit extracted data before export
- Export to Excel format with combined data from all pages
- Automatic file cleanup after processing

## Setup

### Prerequisites

- Python 3.8+
- Typhoon API key (for OCR transcribing + tabular format generatation)
- Poppler (for PDF to image conversion)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdftotable
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Poppler (required for pdf2image):
   - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your TYPHOON API key:
```
TYPHOON_API_KEY=your_typhoon_api_key_here
```

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open the web application in your browser
2. Upload a PDF file containing Thai construction quotes/invoices
3. The system will process each page and extract table data
4. Review and edit the extracted data if needed
5. Validate the combined data from all pages
6. Download the Excel file with the processed data

## Data Schema

The application expects tables with the following structure:
- **Description**: Overall description of the quote
- **Items**: Array of line items with:
  - Order number
  - Item name
  - Quantity and unit
  - Material unit price and total
  - Labor unit price and total
  - Grand total

## File Structure

```
pdftotable/
├── app.py              # Main Flask application
├── templates/
│   ├── index.html      # Upload form
│   ├── review.html     # Data review interface
│   └── validate.html   # Final validation page
├── uploads/            # Temporary file storage
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This file
```

## Security Notes

- API keys are loaded from environment variables
- Uploaded files are automatically cleaned up after processing
- Files older than 1 hour are removed on startup
- Only PDF files are accepted for upload

## License

This project is open source and available under the MIT License.