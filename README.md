# SmartTalk

A fast, modern coding interview application built with **React** and **FastAPI**.

## Features

- **Fast** - React frontend with FastAPI backend (much faster than Streamlit)
- **4 Difficulty Levels** - Easy, Medium, Hard, Expert
- **Test Cases** - Add and run your own test cases
- **AI Scoring** - Gemini AI analyzes and scores your solutions
- **Auto-Generation** - Problems generate in the background
- **Progress Tracking** - Track your quiz progress and scores

## Project Structure

```
smarttalk/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # API key (create this)
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── index.js
│   │   └── App.js           # React app
│   └── package.json
├── run.py                   # Cross-platform runner
├── run.sh                   # Linux/Mac runner
├── run.bat                  # Windows runner
├── .gitignore
└── README.md
```

## Quick Start

### 1. Set up API Key

Create `backend/.env`:

```
GOOGLE_API_KEY=your-api-key-here
```

Get your key from: https://aistudio.google.com/app/apikey

### 2. Run the App

**Option A: Python runner (recommended)**
```bash
python run.py
```

**Option B: Shell script (Linux/Mac)**
```bash
chmod +x run.sh
./run.sh
```

**Option C: Batch file (Windows)**
```cmd
run.bat
```

**Option D: Manual start**

Terminal 1 (Backend):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Terminal 2 (Frontend):
```bash
cd frontend
npm install
npm start
```

### 3. Open the App

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pool/status` | GET | Get problem pool status |
| `/pool/generate` | POST | Generate more problems |
| `/quiz/start` | GET | Start a new quiz (get 4 problems) |
| `/quiz/submit` | POST | Submit solution for scoring |
| `/quiz/run-tests` | POST | Run test cases |
| `/generator/start` | POST | Start background generator |
| `/generator/stop` | POST | Stop background generator |
| `/pool/clear` | POST | Clear the problem pool |

## Tech Stack

- **Frontend**: React 18, Monaco Editor, Tailwind CSS, Axios
- **Backend**: FastAPI, Pydantic, Google Gemini AI
- **Code Editor**: Monaco Editor (same as VS Code)

## Troubleshooting

### "GOOGLE_API_KEY not set"
Create `backend/.env` with your API key.

### "No problems available"
Wait for the background generator to create problems, or manually generate:
```bash
curl -X POST http://localhost:8000/pool/generate
```

### Frontend won't start
Make sure Node.js is installed:
```bash
node --version  # Should be v16+
npm --version   # Should be v8+
```

### Backend errors
Check Python version and dependencies:
```bash
python --version  # Should be 3.9+
pip install -r backend/requirements.txt
```

## License

MIT
