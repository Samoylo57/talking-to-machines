# Talking to Machines Platform (Beta Release)


The Talking to Machines platform is developed to facilitate the design, conduct, and analysis of large-scale experimental trials and treatments through LLM-powered agents. 

---

## ✨ Features

* **CLI‑first Workflow** – build and run large-scale experimental trials straight from your terminal.
* **Python API** – the same engine is importable as a Python package in notebooks and pipelines.
* **Multi‑model Provider** – ships with wrappers for **OpenAI** chat models and the **Hugging Face** Inference API.
* **Reproducible** – every run is JSON‑logged for auditable, reproducible results.

---

## 🔧 Installation

> **Requires Python 3.10 or higher**

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install the talkingtomachines package
```bash
pip install talkingtomachines
```

### 3. Provide API keys

Approach 1: Create a `.env` file in your root directory
```ini
# .env file
OPENAI_API_KEY=sk-...
HF_API_KEY=hf_...
``` 

Approach 2: Export as environment variables 
```bash
export OPENAI_API_KEY=sk-...
export HF_API_KEY=hf_...
````

---

## 🚀 Usage

### CLI Tool

The platform can be used as a CLI tool for non-technical users or users who are not familiar with Python. For this approach, you will need to populate a prompt template to define your experimental setup. Detailed instructions on how to properly set up a prompt template for your experiment can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md)

```bash
# Show all options
$ talkingtomachines --help

# Parse a prompt template containing the experimental setup
$ talkingtomachines path/to/prompt/template.xlsx
```

The CLI displays the details of the experimental setup parsed from the prompt template for your verification and waits for your input:

* input `test` – runs experiment in **TEST** mode (one randomly selected session per treatment).
* input `full` – runs the **FULL** experiment
* input anything else – terminates experiment immediately.

Experimental results are saved to your root directory as a JSON file (`experiment_results/<experiment_id>.json`) and CSV file (`experiment_results/<experiment_id>.csv`).

### Python Package

The platform can also be imported as a Python package for power users/developers who are interested in creating more advanced experimental designs that are currently not supported by the prompt template.

```python
import talkingtomachines
```

---

## 📄 Prompt Template Setup

Detailed instructions on how to populate the prompt template can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md)

You may also check out [`Prompt Template Sample`](https://github.com/talking-to-machines/talking-to-machines/blob/main/demos/public_good_experiment_prompt_template_sample.xlsx) for a minimal starter.

---

## 📜 License

MIT License – see [`LICENSE`](https://github.com/talking-to-machines/talking-to-machines/blob/main/LICENSE) for details.
