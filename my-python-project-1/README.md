# Applicant Tracking System (ATS)

This project is an Applicant Tracking System (ATS) built using Python and Streamlit. It allows users to upload resumes in PDF format and analyze them against job descriptions using the Gemini API.

## Project Structure

```
my-python-project
├── ATS.py          # Main application logic for the ATS
├── venv            # Virtual environment containing dependencies
├── .env            # Environment variables for configuration
└── README.md       # Project documentation
```

## Features

- Upload resumes in PDF format.
- Analyze resumes against job descriptions.
- Get professional evaluations and percentage match results.

## Requirements

- Python 3.x
- Streamlit
- langchain_google_genai
- pdf2image
- Pillow

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-python-project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   Create a `.env` file in the project root and add your API keys and other configuration settings.

6. **Run the application:**
   ```
   streamlit run ATS.py
   ```

## Usage

- Open the application in your web browser.
- Upload a resume and enter the job description.
- Click on the buttons to analyze the resume and get insights.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.