# Friday Demo Script - Week 1: FreshMind Project Foundation

This script is structured for a 5-minute walkthrough presentation of the project's foundation.

---

## 1. Project Introduction & Business Problem (1 Minute)
*   **Slide / Screen:** README.md or Slide 1
*   **Speaker Notes:**
    > "Hello everyone, today I am presenting the foundation of **FreshMind**, our predictive supply chain demand forecasting and replenishment engine.
    > Currently, our fictional FMCG chain, FreshKart, which spans 500 stores, uses a simple 90-day moving average to decide order replenishment.
    > Because this heuristic doesn't capture promotions, events, or seasonality, we face two critical business issues:
    > First, empty shelves and stockouts cause **₹4 Crore per month in lost sales**.
    > Second, over-stocking results in an **8% wastage of perishable goods**.
    > Our goal is to build a machine learning pipeline that forecasts daily demand at the Store-SKU-Day level and recommends order quantities to balance stockout costs and holding costs."

---

## 2. System Architecture & Tech Stack (1 Minute)
*   **Slide / Screen:** docs/architecture.md
*   **Speaker Notes:**
    > "To address this, we designed a modular, production-ready system architecture.
    > The user interacts with a Streamlit Dashboard, which queries a FastAPI backend. This API triggers inference on our Forecast Engine and coordinates with our Inventory Recommendation Engine.
    > Our tech stack is chosen for maximum reliability: Python 3.12, Pandas, and NumPy handle data; LightGBM and PyTorch run our forecasts; and MLflow handles model versioning.
    > By decoupling the forecasting pipeline from the inventory engine, we can change modeling techniques or replenishment policies independently without refactoring the core codebase."

---

## 3. Repository Walkthrough & Configuration (1 Minute)
*   **Slide / Screen:** Directory tree in terminal or file explorer
*   **Speaker Notes:**
    > "Let's take a tour of the repository. We initialized a clean skeleton using standard Python packaging practices.
    > The `src/` directory contains our modular production modules like `data_loader.py` and custom logging utilities.
    > `configs/` isolates configuration parameters into `config.yaml`, ensuring that paths and schemas are never hardcoded.
    > `notebooks/` contains our exploratory work, `tests/` houses our test suite, and `reports/` is where automated profiling reports are saved.
    > We also set up a `.gitignore` that keeps our git history lightweight by ignoring large data files and model binaries, while using `.gitkeep` files to keep folders tracked."

---

## 4. Ingestion & Validation Pipeline Demo (1 Minute)
*   **Slide / Screen:** Run `python src/data/data_loader.py` in terminal, or show reports/data_status_report.md
*   **Speaker Notes:**
    > "Here is our dataset manager in action. Since raw M5 dataset files are around 300MB, we wrote a `download_data.py` script.
    > For fast local setups, running it with the `--sample` flag generates schema-accurate synthetic files.
    > When we execute our validation pipeline, it automatically:
    > 1. Loads calendar, price, and sales files.
    > 2. Performs memory optimization, reducing memory usage by over 80% via integer and float downcasting.
    > 3. Enforces data constraints, ensuring no negative sales, invalid prices, or missing columns.
    > 4. Generates an automated Markdown report.
    > As you can see, all ingestion checks passed successfully."

---

## 5. Next Steps & Roadmap (1 Minute)
*   **Slide / Screen:** Future roadmap in README.md
*   **Speaker Notes:**
    > "Looking ahead, Week 2 is dedicated to Exploratory Data Analysis and building baseline statistical models like Seasonal Naive and ETS.
    > Week 3 will introduce feature engineering and LightGBM models, leading up to deep learning with Temporal Fusion Transformers in Week 4 and full inventory policy optimization in Week 6.
    > This concludes our Week 1 demo. I'm happy to take any questions."
