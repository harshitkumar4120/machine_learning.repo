import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data={
    'applicant':['A','B','C'],
    'education':['Bachlor','master','PHD'],
    'Experiance_years':[2,5,7]
}
df=pd.DataFrame(data)

encoder=OneHotEncoder(sparse_output=False)

encoded=encoder.fit_transform(df[['education']])

encoded_df=pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['education']))

result=pd.concat([df[['applicant','Experiance_years']], encoded_df])
print(result)