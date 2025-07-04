# Customer Churn Analysis Tool

## Overview

This is a Streamlit-based web application designed for comprehensive customer churn analysis. The application provides an end-to-end solution for analyzing customer data, building machine learning models to predict churn, and gaining insights into customer behavior patterns. It features a multi-page architecture with distinct sections for data upload, exploratory analysis, model training, and predictions.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Multi-page application with sidebar navigation
- **UI Components**: Interactive widgets, charts, and data displays
- **State Management**: Streamlit session state for data persistence across pages

### Backend Architecture
- **Core Logic**: Python-based data processing and machine learning pipeline
- **Data Processing**: Pandas for data manipulation and preprocessing
- **Machine Learning**: Scikit-learn ecosystem with multiple algorithm support
- **Visualization**: Plotly for interactive charts and Seaborn/Matplotlib for statistical plots

### Modular Design
- **Utils Package**: Reusable components for data processing, model training, and visualizations
- **Page Structure**: Separate files for each major functionality (upload, analysis, training, predictions)
- **Object-Oriented Components**: Dedicated classes for data processing, model training, and visualization

## Key Components

### 1. Data Processing Module (`utils/data_processor.py`)
- **DataProcessor Class**: Handles comprehensive data preprocessing pipeline
- **Features**: Missing value imputation, categorical encoding, feature scaling, duplicate removal
- **Preprocessing Steps**: StandardScaler for numerical features, LabelEncoder for categorical variables
- **Data Validation**: Automatic data type detection and handling

### 2. Model Training Module (`utils/model_trainer.py`)
- **ModelTrainer Class**: Manages multiple machine learning algorithms
- **Supported Models**: Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **Cross-Validation**: Built-in support for model validation

### 3. Visualization Module (`utils/visualizations.py`)
- **ChurnVisualizer Class**: Creates interactive charts and plots
- **Chart Types**: Distribution plots, box plots, correlation heatmaps, ROC curves
- **Interactive Features**: Plotly-based interactive visualizations
- **Color Scheme**: Consistent visual theme across all charts

### 4. Application Pages
- **Main App (`app.py`)**: Landing page with application overview and navigation
- **Data Upload (`pages/01_Data_Upload.py`)**: File upload, data preview, and preprocessing
- **Exploratory Analysis (`pages/02_Exploratory_Analysis.py`)**: Data visualization and statistical analysis
- **Model Training (`pages/03_Model_Training.py`)**: ML model training and evaluation
- **Predictions (`pages/04_Predictions.py`)**: Single customer and batch prediction capabilities

## Data Flow

1. **Data Ingestion**: Users upload CSV files containing customer data
2. **Data Preprocessing**: Automatic cleaning, encoding, and scaling of features
3. **Exploratory Analysis**: Statistical analysis and visualization of data patterns
4. **Feature Engineering**: Preparation of features for machine learning models
5. **Model Training**: Training multiple ML algorithms with cross-validation
6. **Model Evaluation**: Comprehensive performance metrics and comparison
7. **Prediction Generation**: Single customer or batch predictions using trained models
8. **Results Visualization**: Interactive charts and model performance displays

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Machine learning algorithms and preprocessing tools

### Machine Learning
- **XGBoost**: Gradient boosting framework for enhanced model performance
- **Scikit-learn**: Comprehensive ML library for classification algorithms

### Visualization
- **Plotly**: Interactive plotting library for dynamic charts
- **Seaborn**: Statistical data visualization
- **Matplotlib**: Base plotting library for static charts

### Data Processing
- **Pandas**: Primary data manipulation library
- **NumPy**: Mathematical operations and array handling

## Deployment Strategy

### Local Development
- **Environment**: Python 3.7+ with pip package management
- **Setup**: Requirements installation and Streamlit server execution
- **Development Server**: `streamlit run app.py` for local testing

### Production Considerations
- **Scalability**: Session state management for multi-user environments
- **Data Storage**: In-memory storage suitable for small to medium datasets
- **Performance**: Efficient data processing for real-time predictions
- **Security**: File upload validation and data sanitization

### Future Enhancements
- **Database Integration**: Potential PostgreSQL integration for persistent data storage
- **API Endpoints**: REST API development for external system integration
- **Cloud Deployment**: Containerization and cloud platform deployment options


## User Preferences

Preferred communication style: Simple, everyday language.
