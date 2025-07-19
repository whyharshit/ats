# 📄 ATS (Applicant Tracking System) Expert

An intelligent resume analysis tool that uses Google's Gemini AI to evaluate resumes against job descriptions and provide detailed feedback.

## 🚀 Features

- **Resume Analysis**: Upload PDF resumes and get detailed professional evaluation
- **ATS Compatibility Check**: Get percentage match with job descriptions
- **Keyword Analysis**: Identify missing keywords and strengths/weaknesses
- **Visual Interface**: Clean Streamlit-based web interface
- **AI-Powered**: Uses Google Gemini 1.5 Flash for intelligent analysis

## 📋 Prerequisites

Before running this application, you need to install:

1. **Python 3.8+**
2. **Poppler for Windows**: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Extract to `C:\poppler-24.08.0\Library\bin` (or update the path in your `.env` file)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd chatmodels
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env` (if available)
   - Add your Google API key to `.env`:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     POPPLER_PATH=C:\poppler-24.08.0\Library\bin
     ```

## 🔑 Getting Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🚀 Usage

1. **Run the application**
   ```bash
   streamlit run ATS.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter job description** in the text area

4. **Upload your resume** (PDF format only)

5. **Choose analysis type**:
   - **"Tell Me About the Resume"**: Get detailed professional evaluation
   - **"Percentage Match"**: Get ATS compatibility score and keyword analysis

## 📁 Project Structure

```
chatmodels/
├── ATS.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── venv/                # Virtual environment (not in git)
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key
- `POPPLER_PATH`: Path to Poppler bin directory

### Customizing Poppler Path

If you installed Poppler in a different location, update the `POPPLER_PATH` in your `.env` file:

```
POPPLER_PATH=C:\your\poppler\path\bin
```

## 📝 Features in Detail

### Resume Analysis
- Professional evaluation by HR perspective
- Strengths and weaknesses identification
- Alignment assessment with job requirements

### ATS Compatibility
- Percentage match calculation
- Missing keywords identification
- ATS scanner simulation
- Concluding thoughts and recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

- **API Key Security**: Never commit your `.env` file to version control
- **PDF Processing**: Ensure Poppler is properly installed for PDF processing
- **File Size**: Large PDF files may take longer to process
- **Internet Connection**: Required for Google Gemini API calls

## 🐛 Troubleshooting

### Common Issues

1. **Poppler not found**: Ensure Poppler is installed and path is correct in `.env`
2. **API Key error**: Verify your Google API key is valid and has sufficient credits
3. **PDF upload fails**: Ensure the file is a valid PDF and not corrupted

### Support

If you encounter any issues, please:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure environment variables are set correctly
4. Open an issue on GitHub with detailed error information

## 🔮 Future Enhancements

- [ ] Support for multiple resume formats
- [ ] Batch processing capabilities
- [ ] Custom scoring algorithms
- [ ] Integration with job boards
- [ ] Resume optimization suggestions
- [ ] Export analysis reports 