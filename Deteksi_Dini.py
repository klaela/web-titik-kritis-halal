import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC

st.title('Website Prediksi Titik Kritis Halal Bahan Hewani')

st.caption('Website ini merupakan bagian dari penelitian tugas akhir Penelurusan Titik Kritis Halal, Teknik Informatika, Institut Teknologi Sepuluh Nopember, Surabaya, 2022')

uploaded_file = st.file_uploader("Choose a log event file")

data = pd.read_csv("./app_history.csv", sep=';',  error_bad_lines=False)  # read a CSV file inside the 'data" folder next to 'app.py'
#st.dataframe(data)
data = dataset = data.pivot_table(index="CaseID", columns="Activity", values="Status_Halal", aggfunc='first')
column =data.columns
#st.dataframe(column)

dataset = dataset[dataset['Ambil Kesimpulan'].isna() == False]
feature = dataset.loc[:, dataset.columns != "Ambil Kesimpulan"]
label = dataset["Ambil Kesimpulan"]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(feature)
feature_enc = enc.transform(feature)

#prediksi
# cv_method = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
# params_RF = {"min_samples_split": [2, 6, 20],
#         "n_estimators" :[100,200,300],
#         "criterion": ["gini", "entropy"]             
#         }

# GridSearchCV_RF = GridSearchCV(estimator=RandomForestClassifier(), 
#                           param_grid=params_RF, 
#                           cv=cv_method,
#                           verbose=1, 
#                           n_jobs=3,
#                           scoring="accuracy", 
#                           return_train_score=True
#                           )
# GridSearchCV_RF.fit(feature_enc,label);

param_grid = {'C': [10],
    'gamma': [1, 0.1, 0.01, 0.001],
             'kernel': ['poly', 'rbf', 'sigmoid']}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True,verbose=2)
grid_fit = grid.fit(feature_enc,label)

if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     #st.write(bytes_data)

     # To convert to a string based IO:
     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
     #st.write(stringio)

     # To read file as string:
     string_data = stringio.read()
     #st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
     df = pd.read_csv(uploaded_file, sep=';')
     #st.dataframe(df)
    
     #preprocesing data
     test_data = pd.DataFrame(columns = column)

     df_test = df.pivot_table(index="CaseID", columns="Activity", values="Status_Halal", aggfunc='first')
     #st.dataframe(df_test)

     test_data=test_data.append(df_test)
     #st.dataframe(test_data)

     test_data = test_data.loc[:, test_data.columns != "Ambil Kesimpulan"]

    #  pipeline_rf = Pipeline([('enc', enc), ('GridSearchCV_RF', GridSearchCV_RF)])
    #  hasil_rf = pipeline_rf.predict(test_data)

     pipeline_svm = Pipeline([('enc', enc), ('grid', grid)])
     hasil_svm = pipeline_svm.predict(test_data)

     st.write('Hasil Deteksi:')
     #kolom = { 'CaseID': test_data.index, 'Skenario': [4,2,3,1,5], 'Hasil': hasil_svm}
     kolom = { 'CaseID': test_data.index, 'Hasil': hasil_svm}
     df_ujicoba = pd.DataFrame(kolom)
     df_ujicoba['Keterangan'] = ["Bahan termasuk bahan halal" if x == 'Halal' else "Bahan berpotensi haram" for x in df_ujicoba["Hasil"]]
     #st.dataframe(df_ujicoba.sort_values(by="Skenario", ignore_index=True))
     st.dataframe(df_ujicoba)

     st.warning('Penyebab produk berpotensi haram:')
     potensi = df.loc[df['Status_Halal'] == 'Haram', 'Activity']
     if potensi is not None:
        #st.dataframe(potensi)
        st.tabel(potensi)