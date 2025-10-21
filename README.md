# Talking to Machines Platform (Beta Release)


The **talkingtomachines** platform is developed to facilitate the design, conduct, and analysis of large-scale experimental trials and treatments through LLM-powered agents. 

---

## ✨ Features

* **CLI‑first Workflow** – build and run large-scale experimental trials straight from your terminal.
* **Python API** – the same engine is importable as a Python package in notebooks and pipelines.
* **Multi‑model Provider** – ships with wrappers for **OpenAI** chat models and the **Hugging Face** Inference API.
* **Reproducible** – every run is JSON‑logged for auditable and reproducible results.

---

## 🔧 Installation


### 1. Install Miniconda

* Download **Miniconda** for your OS (Windows, macOS):
  [https://www.anaconda.com/docs/getting-started/miniconda/install](https://www.anaconda.com/docs/getting-started/miniconda/install)
* Run the installer and accept the defaults.

  * On Windows, allow the installer to add Conda to your PATH.
  * On macOS, you may need to restart your terminal after installation.

Verify your installation by running the following command in your terminal (for macOS) or Windows Powershell (for Windows):

```bash
conda --version  # Displays conda's version number
```


### 2. Create and activate a Conda environment

Create a fresh environment (Python 3.10 or newer):

```bash
conda create -n your-env-name python=3.10
```

Activate your newly created conda environment:

```bash
conda activate your-env-name
```

Upgrade `pip` inside the environment (optional but recommended):

```bash
python -m pip install --upgrade pip
```

Verify that `python` and `pip` are properly installed:
```bash
python --version   # should print 3.10.x or higher
pip --version
```

### 3. Install the talkingtomachines package
```bash
pip install \
 --index-url https://test.pypi.org/simple/ \
 --extra-index-url https://pypi.org/simple \
 talkingtomachines
```

### 4. Provide API keys for OpenAI and Hugging Face

Export the API keys for OpenAI and Hugging Face as environment variables on your terminal (for macOS) or Windows Powershell (for Windows):
```bash
export OPENAI_API_KEY=sk-...
export HF_API_KEY=hf_...
````

---

## Video Walkthrough
A video walkthrough on how to set up the Talking To Machines platform for both macOS and Windows devices can be found here:

macOS: [Video Walkthrough](https://www.loom.com/share/a2c15f1258d5436eaeca197998286cd9?sid=7958bdbe-2f34-4d5f-8d47-49fd49cb315c)

Windows: [Video Walkthrough](https://www.loom.com/share/79969b38be6d4c2387d19ecc3e54ae4d?sid=d372a5e7-6dd5-4923-b838-69eba7dce20a)


---

## 🚀 Usage

### CLI Tool

The platform can be used as a CLI tool for non-technical users or users who are not familiar with Python. For this approach, you will need to populate a prompt template to define your experimental setup. Detailed instructions on how to properly set up a prompt template for your experiment can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md)

```bash
# Show all options
$ talkingtomachines --help

# Provide the file path to the prompt template to parse the experimental setup
$ talkingtomachines path/to/prompt/template.xlsx
```

The CLI displays the details of the experimental setup parsed from the prompt template for your verification and waits for your input:

* input `test` – runs experiment in **TEST** mode (one randomly selected session per treatment).
* input `full` – runs the **FULL** experiment
* input anything else – terminates experiment immediately.

Experimental results are saved to your root directory as a JSON file containing the raw outputs (`experiment_results/<experiment_id>.json`) and a CSV file containing the formatted outputs (`experiment_results/<experiment_id>.csv`).

### Python Package

The platform can also be imported as a Python package for power users/developers who are interested in creating more advanced experimental designs that are currently not supported by the prompt template.

```python
import talkingtomachines
```

Note: A simple, end‑to‑end Python example on how to construct and conduct an experiment using the talkingtomachines package will be added in the future.

---

## 📄 Prompt Template Setup

Detailed instructions on how to populate the prompt template can be found here: [`Prompt Template Instructions`](https://github.com/talking-to-machines/talking-to-machines/tree/main/talkingtomachines/interface/README.md)

You may also check out [`Demo Example`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment) to see an example of a prompt template that was prepared by the development team to replicate a public goods experiment using the **talkingtomachines** platform, or a clean version of the prompt template for creating new experiments [`New Prompt Template`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/prompt_template.xlsx).

---

## 📜 License

MIT License – see [`LICENSE`](https://github.com/talking-to-machines/talking-to-machines/blob/main/LICENSE) for details.
