eak-eexc-idu
Awoleye Goodness
20:49
🏥 Health Price Estimator (End-to-End ML Pipeline & Streamlit App)
A production-ready machine learning web application designed to predict healthcare costs and medical service pricing based on patient profiles and medical attributes. This project showcases a complete end-to-end data science pipeline—from exploratory data analysis and preprocessing to robust model persistence and interactive web deployment.
🚀 Live Demo
(Optional: Add your live Streamlit Community Cloud app link here once deployed!)
🔗 View Live App
🛠️ Tech Stack & Libraries
 * Python 3.10+
 * Pandas & NumPy: Data wrangling and numerical processing.
 * Scikit-Learn: Machine learning pipelines (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, RandomForestRegressor).
 * Joblib: Model persistence and serialization (.pkl).
 * Streamlit: Interactive web interface framework.
📊 Project Workflow
 * Data Cleaning & Preprocessing: Cleaning column headers (.str.strip()), managing missing health data, and handling categorical variables.
 * Pipelines & ColumnTransformer: Bundling numerical feature scaling (StandardScaler) and text encoding (OneHotEncoder) into an automated pipeline to completely prevent data leakage.
 * Model Training: Training an ensemble RandomForestRegressor (100 estimators, random_state=42) on an 80/20 train-test split to predict continuous medical costs accurately.
 * Model Export: Saving the entire production-ready pipeline into a binary file (health_model.pkl) using joblib.
 * Web Deployment: Building a user-friendly interface using Streamlit to accept live patient inputs and output instant health cost estimates.
📂 Project Directory Structure
health-price-estimator/
│
├── health_data.csv        # Raw dataset
├── health_model.pkl       # Exported trained Scikit-Learn pipeline
├── app.py                 # Streamlit web application script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

⚙️ How to Run Locally
To run this project on a local machine, follow these steps:
 * Clone the repository:
   git clone https://github.com/your-username/health-price-estimator.git
cd health-price-estimator

 * Install dependencies:
   pip install -r requirements.txt

 * Run the Streamlit app:
   streamlit run app.py

📝 Requirements File (requirements.txt)
pandas
numpy
scikit-learn
joblib
streamlit

👨‍💻 Author
Jomiloju
Student & Machine Learning Developer
