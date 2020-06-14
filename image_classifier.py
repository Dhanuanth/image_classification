import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=pd.read_csv('mnist_train.csv')
print(data.head())
a=data.iloc[69,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
plt.show()
x_pxl=data.iloc[:,1:]
x_label=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(x_pxl,x_label,test_size=0.2,random_state=4)
print(y_train.head())

rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count+=1

print(count,count/len(pred))
