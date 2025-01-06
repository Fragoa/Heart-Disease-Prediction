import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler



base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir,'data','raw')
file_path = os.path.join(data_dir,"Heart_disease_cleveland.csv")
df = pd.read_csv(file_path)

print("Original DataFrame:")
print(df)


categorical_columns = ['exang','fbs','sex','ca','cp','restecg','thal','slope']
numerical_columns = ['age','trestbps','chol','thalach','oldpeak']


# OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encodedData = encoder.fit_transform(df[categorical_columns])
encodedDf = pd.DataFrame(encodedData, columns=encoder.get_feature_names_out(categorical_columns))

# StandardScaler
scaler = StandardScaler()
scaledData = scaler.fit_transform(df[numerical_columns])
scaledDf = pd.DataFrame(scaledData, columns=numerical_columns)

# Combine both encoded and scaled data into a final DataFrame
dfFinal = pd.concat([df.drop(columns=categorical_columns), encodedDf, scaledDf], axis=1)



preprocessing_dir = os.path.join(base_dir, 'data', 'processed')
if not os.path.exists(preprocessing_dir):
    os.makedirs(preprocessing_dir)
processed_file_path = os.path.join(preprocessing_dir, "Heart_disease_processed.csv")
dfFinal.to_csv(processed_file_path, index=False)

