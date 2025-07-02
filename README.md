# 🧠 Interpreter-GPT

**Interpreter-GPT** is a Python-based tool that leverages OpenAI's GPT models to interpret and summarize structured episode data. It reads data from a CSV file and uses natural language generation to extract insights or generate summaries — useful for researchers, content creators, and data analysts.

## 📁 Project Structure

- `main.py` – Core script for loading data and generating interpretations using GPT.
- `episode_info.csv` – Source file containing episode metadata or descriptions.
- `qrcodes/` – (Optional) Folder for storing QR codes or image outputs.
- `Pipfile` / `Pipfile.lock` – Dependency and environment management files.

## ✨ Features

- 🔍 Natural language summaries from structured data.
- 🤖 Integration with OpenAI's GPT API.
- 🧾 Customizable prompts and logic.
- 📦 Lightweight and easy to adapt to other CSV-based inputs.

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SarinaHamedani/interpreter-gpt.git
   cd interpreter-gpt
2. **Activate the virtual environment**:
    ```bash
    pipenv shell
3. **Install dependencies (requires Pipenv)**:
    ```bash
    pipenv install
