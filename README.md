# Spam Detector

A machine learning project for detecting spam messages using Python. This repository contains scripts for data cleaning, inspection, model training, and an application interface, along with data and exploratory notebooks.

---

## Demo

Below are screenshots of the Email/SMS Spam Detector web application in action:

### Example 1: Spam Message Detection

![image1](image1)

*The interface analyzes a suspicious message and predicts it as **SPAM** with 96.40% confidence. It highlights that the message contains financial terms, but no URL or urgent language.*

---

### Example 2: Not Spam Message Detection

![image2](image2)

*Here, a regular transactional message is analyzed and predicted as **NOT SPAM** with 24.05% confidence. The message contains a URL but neither urgent language nor financial terms.*

---

## Directory Structure

```
.
├── app/                # Application front-end (HTML and supporting files)
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── clean_data.py       # Script to clean and preprocess data
├── inspect_data.py     # Script for data exploration and inspection
├── requirements.in     # Python dependencies (input format)
├── requirements.txt    # Python dependencies (full list)
├── run.py              # Script to launch the application
├── train_model.py      # Script to train ML models
```

## Features

- **Data Cleaning:** Automated scripts to clean and preprocess raw data (`clean_data.py`).
- **Data Inspection:** Tools for understanding dataset characteristics (`inspect_data.py`).
- **Model Training:** End-to-end model training pipeline (`train_model.py`).
- **App Interface:** All-in-one application in the `app/` folder, with `run.py` to launch.
- **Jupyter Notebooks:** Notebooks for EDA and prototyping in `notebooks/`.
- **Reproducibility:** All dependencies specified in `requirements.in` and `requirements.txt`.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dwdjitendra-cloud/spam-detector.git
   cd spam-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Clean and inspect data:**
   ```bash
   python clean_data.py
   python inspect_data.py
   ```

4. **Train the model:**
   ```bash
   python train_model.py
   ```

5. **Run the app:**
   ```bash
   python run.py
   ```

## Project Structure Overview

- **`app/`**: Contains the web or local application interface code.
- **`data/`**: Place your datasets here.
- **`notebooks/`**: Jupyter notebooks for data exploration and research.
- **`clean_data.py`**: Data cleaning logic.
- **`inspect_data.py`**: Data visualization and basic statistics.
- **`train_model.py`**: Model training and evaluation.
- **`run.py`**: Main entry point for running the app.
- **`requirements.*`**: Lists of Python dependencies.

## License

This project is provided for educational purposes. Please see the repository for licensing details.

---

**Author:** [dwdjitendra-cloud](https://github.com/dwdjitendra-cloud)