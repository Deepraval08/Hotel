# 🏨 Predicting Hotel Reservation Cancellations

## 📌 Project Overview

This project focuses on building a **machine learning–based system** to predict whether a hotel reservation is likely to be **cancelled or not**. Accurate cancellation prediction helps hotels take proactive actions to reduce revenue loss, manage resources efficiently, and improve customer experience.

The solution includes:

* A complete **ML pipeline** for training and prediction
* **FastAPI** for serving the model as an API
* **Streamlit** for an interactive frontend
* **Docker** for environment consistency and deployment

---

## 🎯 Problem Statement

Hotel reservation cancellations significantly impact revenue and operational planning. This project uses historical booking data and machine learning models to predict cancellations in advance, enabling smarter business decisions.

---

## 🚀 Project Benefits

### 💰 Revenue Optimization

* Predict potential cancellations early
* Enable dynamic pricing or controlled overbooking
* Reduce revenue loss

### 🧑‍💼 Resource Management

* Better staff scheduling
* Improved room and inventory planning

### 📊 Customer Insights

* Identify key factors influencing cancellations
* Improve customer satisfaction and retention strategies

---

## 🛠️ Tech Stack

* **Programming Language:** Python 🐍
* **Machine Learning:** Scikit-learn (ML Pipeline)
* **Backend API:** FastAPI ⚡
* **Frontend:** Streamlit 🎨
* **Environment & Deployment:** Docker 🐳

---

## ⚙️ How to Run the Project Locally

Follow the steps below to clone and run the project on your local machine.

### 1️⃣ Create a Virtual Environment

bash
python -m venv venv

Activate the virtual environment:

* **Windows**

venv\Scripts\activate

* **Linux / macOS**

source venv/bin/activate


### 2️⃣ Install Required Packages

Install all dependencies listed in `requirements.txt`:

pip install -r requirements.txt

### 3️⃣ Run the ML Pipeline / Main Script

Execute the main Python file (for training or prediction logic):

python run.py


### 4️⃣ Run the Streamlit Application

Launch the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

Once started, open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

### 5️⃣ Run the FastAPI Server

Start the FastAPI backend using Uvicorn:

bash
uvicorn api:app --reload

FastAPI will be available at:

* API: `http://127.0.0.1:8000`
* Swagger Docs: `http://127.0.0.1:8000/docs`


## 🐳 Run Using Docker (Optional)

If Docker is configured:


docker build -t hotel-cancellation-predictor .
docker run -p 8000:8000 hotel-cancellation-predictor


## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.


## 📬 Contact

For questions or suggestions, feel free to connect.

⭐ If you find this project useful, don’t forget to star the repository!


