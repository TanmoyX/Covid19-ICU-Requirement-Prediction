import pandas as pd

# Pre-processing library
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class classifier:
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

    def fit_rf_classifier(self):
        covid_hos = pd.read_csv("covid.csv", low_memory = False)
        # Cleaning the data for abnormal values
        covid_hos = covid_hos.loc[(covid_hos.patient_type == 1)  | (covid_hos.patient_type == 2)]
        covid_hos = covid_hos.loc[(covid_hos.pneumonia == 1)  | (covid_hos.pneumonia == 2)]
        covid_hos = covid_hos.loc[(covid_hos.diabetes == 1)  | (covid_hos.diabetes == 2)]
        covid_hos = covid_hos.loc[(covid_hos.copd == 1)  | (covid_hos.copd == 2)]
        covid_hos = covid_hos.loc[(covid_hos.asthma == 1)  | (covid_hos.asthma == 2)]
        covid_hos = covid_hos.loc[(covid_hos.inmsupr == 1)  | (covid_hos.inmsupr == 2)]
        covid_hos = covid_hos.loc[(covid_hos.cardiovascular == 1)  | (covid_hos.cardiovascular == 2)]
        covid_hos = covid_hos.loc[(covid_hos.obesity == 1)  | (covid_hos.obesity == 2)]
        covid_hos = covid_hos.loc[(covid_hos.renal_chronic == 1)  | (covid_hos.renal_chronic == 2)]
        covid_hos = covid_hos.loc[(covid_hos.tobacco == 1)  | (covid_hos.tobacco == 2)]
        covid_hos = covid_hos.loc[(covid_hos.covid_res == 1)  | (covid_hos.covid_res == 2)]
        
        # One-Hot-Encoding of the data
        covid_hos.sex = covid_hos.sex.apply(lambda x: x if x == 1 else 0)
        covid_hos.patient_type = covid_hos.patient_type.apply(lambda x: x if x == 1 else 0)
        covid_hos.pneumonia = covid_hos.pneumonia.apply(lambda x: x if x == 1 else 0)
        covid_hos.diabetes = covid_hos.diabetes.apply(lambda x: x if x == 1 else 0)
        covid_hos.copd = covid_hos.copd.apply(lambda x: x if x == 1 else 0)
        covid_hos.asthma = covid_hos.asthma.apply(lambda x: x if x == 1 else 0)
        covid_hos.inmsupr = covid_hos.inmsupr.apply(lambda x: x if x == 1 else 0)
        covid_hos.cardiovascular = covid_hos.cardiovascular.apply(lambda x: x if x == 1 else 0)
        covid_hos.obesity = covid_hos.obesity.apply(lambda x: x if x == 1 else 0)
        covid_hos.renal_chronic = covid_hos.renal_chronic.apply(lambda x: x if x == 1 else 0)
        covid_hos.tobacco = covid_hos.tobacco.apply(lambda x: x if x == 1 else 0)
        covid_hos.covid_res = covid_hos.covid_res.apply(lambda x: x if x == 1 else 0)
        
        # COVID-19 posiitve cases only and then drop the result column
        covid_hos = covid_hos.loc[covid_hos.covid_res == 1]
        covid_hos.drop(columns = ['covid_res', 'icu'], inplace = True)
        
        # Applying scaling on continuous feature
        scaler = MinMaxScaler()
        #covid_hos[['age']] = scaler.fit_transform(covid_hos[['age']])
        
        # X and y creation for training data
        y_hos = covid_hos.pop('patient_type')
        X_hos = covid_hos
        
        # Fitting the model
        self.rf_classifier.fit(X_hos, y_hos)
        
        return self.rf_classifier, scaler
    
    def fit_lr_classifier(self):
        covid_icu = pd.read_csv("covid.csv", low_memory = False)
        
        # Cleaning the data for abnormal values
        covid_icu = covid_icu.loc[(covid_icu.pneumonia == 1)  | (covid_icu.pneumonia == 2)]
        covid_icu = covid_icu.loc[(covid_icu.diabetes == 1)  | (covid_icu.diabetes == 2)]
        covid_icu = covid_icu.loc[(covid_icu.copd == 1)  | (covid_icu.copd == 2)]
        covid_icu = covid_icu.loc[(covid_icu.asthma == 1)  | (covid_icu.asthma == 2)]
        covid_icu = covid_icu.loc[(covid_icu.inmsupr == 1)  | (covid_icu.inmsupr == 2)]
        covid_icu = covid_icu.loc[(covid_icu.cardiovascular == 1)  | (covid_icu.cardiovascular == 2)]
        covid_icu = covid_icu.loc[(covid_icu.obesity == 1)  | (covid_icu.obesity == 2)]
        covid_icu = covid_icu.loc[(covid_icu.renal_chronic == 1)  | (covid_icu.renal_chronic == 2)]
        covid_icu = covid_icu.loc[(covid_icu.tobacco == 1)  | (covid_icu.tobacco == 2)]
        covid_icu = covid_icu.loc[(covid_icu.covid_res == 1)  | (covid_icu.covid_res == 2)]
        covid_icu = covid_icu.loc[(covid_icu.icu == 1)  | (covid_icu.icu == 2)]
        
        # One-Hot-Encoding of the data
        covid_icu.sex = covid_icu.sex.apply(lambda x: x if x == 1 else 0)
        covid_icu.pneumonia = covid_icu.pneumonia.apply(lambda x: x if x == 1 else 0)
        covid_icu.diabetes = covid_icu.diabetes.apply(lambda x: x if x == 1 else 0)
        covid_icu.copd = covid_icu.copd.apply(lambda x: x if x == 1 else 0)
        covid_icu.asthma = covid_icu.asthma.apply(lambda x: x if x == 1 else 0)
        covid_icu.inmsupr = covid_icu.inmsupr.apply(lambda x: x if x == 1 else 0)
        covid_icu.cardiovascular = covid_icu.cardiovascular.apply(lambda x: x if x == 1 else 0)
        covid_icu.obesity = covid_icu.obesity.apply(lambda x: x if x == 1 else 0)
        covid_icu.renal_chronic = covid_icu.renal_chronic.apply(lambda x: x if x == 1 else 0)
        covid_icu.tobacco = covid_icu.tobacco.apply(lambda x: x if x == 1 else 0)
        covid_icu.covid_res = covid_icu.covid_res.apply(lambda x: x if x == 1 else 0)
        covid_icu.icu = covid_icu.icu.apply(lambda x: x if x == 1 else 0)
        
        # COVID-19 posiitve cases only and then drop the result column
        covid_icu = covid_icu.loc[covid_icu.covid_res == 1]
        covid_icu.drop(columns = ['covid_res', 'patient_type'], inplace = True)
        
        train, test = train_test_split(covid_icu, test_size = 0.3, random_state = 100)
        
        
        
        # Applying scaling on continuous feature
        scaler = MinMaxScaler()
        train[['age']] = scaler.fit_transform(train[['age']])
        
        # X and y creation for training data
        y_icu = train.pop('icu')
        X_icu = train
        
        sm = SMOTE()
        X_icu, y_icu = sm.fit_resample(X_icu, y_icu)
        
        lr_classifier = LogisticRegression()
        lr_classifier.fit(X_icu, y_icu)
        test[['age']] = scaler.transform(test[['age']])
        
        y_test = test.pop('icu')
        x_test = test
        
        pred = lr_classifier.predict(x_test)
        
        cm = confusion_matrix(y_test,pred)
        print(cm)
        
        return lr_classifier, scaler