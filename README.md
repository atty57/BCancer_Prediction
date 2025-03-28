# Breast Cancer Prediction using Machine Learning

## Project Overview
Breast cancer is among the most prevalent cancers affecting millions globally. Early detection plays a critical role in improving patient survival rates. This project employs various machine learning algorithms to predict breast cancer accurately, classifying tumors as malignant or benign based on clinical and demographic data.

## Description
Breast cancer originates when cells in the breast begin to grow uncontrollably, forming lumps or tumors. This project utilizes machine learning methods to develop predictive models that classify breast tumors as malignant or benign, thus aiding early diagnosis and treatment.

## Dataset
The dataset comprises features computed from digitized images of Fine Needle Aspirate (FNA) samples of breast lumps. It consists of 569 records and 32 columns, with the binary target variable indicating benign (`0`) or malignant (`1`) diagnoses.

## Machine Learning Models Used
The following machine learning models have been trained, tested, and evaluated:
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Support Vector Machines (SVM)
- Random Forest

## Data Preprocessing
- **Feature Scaling**: Applied StandardScaler from Scikit-learn.
- **Label Encoding**: Diagnosis was encoded as `Malignant = 1` and `Benign = 0`.
- **Feature Selection**: Relevant features were selected based on their predictive power.

## Hyperparameter Tuning
Hyperparameters of each model were optimized using GridSearchCV with 5-fold cross-validation:

| Model                 | Best Hyperparameters                        | Accuracy (Tuned) | Accuracy (Default) |
|-----------------------|---------------------------------------------|------------------|-----------------------|
| Logistic Regression   | C=0.1, Penalty='l2'                         | 0.9912           | 0.974                 |
| kNN                   | Metric='Euclidean', Neighbors=9             | 0.965            | 0.947                 |
| SVM                   | Kernel='linear', C=0.1                      | 0.9825           | 0.9825                |
| Random Forest         | Estimators=50, Max Depth=6                  | 0.965            | 0.956                 |

## Hyperparameter Tuning
Hyperparameters were optimized using GridSearchCV with 5-fold cross-validation, ensuring the best performance and generalization of the models.

## Experimental Results
The Logistic Regression model achieved the highest accuracy (0.9912) and highest recall, making it particularly suitable for medical diagnostic purposes where reducing false negatives is critical.

## Project Structure
```
.
├── data/
│   └── breast-cancer-data.csv
├── notebooks/
│   └── model-development.ipynb
├── reports/
│   └── project-report.pdf

## Usage
- Clone the repository
- Install necessary libraries using `pip install -r requirements.txt`
- Run the notebook `model-development.ipynb` for detailed execution

## Contributors
- Melvin Oswald Sahaya Anbarasu ([oswald@udel.edu](mailto:oswald@udel.edu))
- Nii Otu Tackie-Otoo ([niiotu@udel.edu](mailto:niiotu@udel.edu))
- Annamalai Muthupalaniappan ([annamala@udel.edu](mailto:annamala@udel.edu))
- Atharva Vichare ([atharvav@udel.edu](mailto:atharvav@udel.edu))

## Acknowledgments
- University of Delaware
- Kaggle Dataset Providers

## References
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Kaggle: https://www.kaggle.com/

