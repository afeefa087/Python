# head
## sub head

### 1. 
```
print("hi")
```

### 2 
[experiment2](https://github.com/drishyats/test/blob/main/hello.py)

3
```import re
pattern='^[a-zA-Z0-9-_]+@+[a-zA-Z0-9]+.[a-z]{3,3}$'
test_string=input('Enter a valid email id ')
result=re.match(pattern,test_string)
if result:
  print("Valid user id")
else:
  print("Invalid user id")
  ```
  
  4
  ```import matplotlib.pyplot as plt
import numpy as np
y=np.array([20,20,10,10,30,10])
plt.pie(y)
plt.show()
```

5
```import matplotlib.pyplot as plt
import numpy as np
y=np.array([25,20,15,40])
mylabels=["Apple","Bananas","Cherries","Dates"]
plt.pie(y,labels=mylabels,startangle=90)
plt.show()
```
6
```import matplotlib.pyplot as plt
import numpy as np
#plot 1:
x=np.array([0,1,2,3])
y=np.array([3,8,1,10])
plt.subplot(1,2,1)
plt.plot(x,y)
#plot 2:
x=np.array([0,1,2,3])
y=np.array([10,20,30,40])
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()
```
7
```import re
string='Twelve 12 Eighty nine:89.'
pattern='\d+'
result=re.split(pattern,string)
print(result)
```
8
```import matplotlib.pyplot as plt
x1=[1,2,3]
y1=[2,4,1]
plt.plot(x1,y1,label="line 1")
x2=[1,2,3]
y2=[2,4,1]
```
9
```import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(1,5)
y=x**3
fig=plt.figure()
plt.plot(x,y,'r')
plt.show()
```
10
```import numpy as np
arr=np.array([1,2,3,4,5])
print(arr)
```
11
```import numpy as np
print(np.__version__)
```
12
```import numpy as np
arr=np.array([1,2,3,4,5])
print(arr[2]+arr[3])
```
13
```import numpy as np
arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print("2nd element on 1st row",arr[0,1])
```
14
```import numpy as np
arr=np.array([1,2,3,4,5,6,7])
print(arr[1:5])
```
15
```import numpy as np
arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr.shape)
```
16
```import numpy as np
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
arr=np.concatenate((arr1,arr2))
print(arr)
```
17
```from matplotlib import pyplot as plt
Names=['Arun','James','Ricky','Patrick']
Marks=[51,87,45,67]
plt.bar(Names,Marks,color='orange')
plt.title('Result')
plt.xlabel('Names')
plt.ylabel('Marks')
plt.show()
```
18
```import matplotlib.pyplot as plt
left=[1,2,3,4,5]
height=[10,24,36,40,5]
tick_label=['one','two','three','four','five']
plt.bar(left,height,tick_label=tick_label,width=0.5,color=['skyblue','lightgreen','pink','darkred','violet'])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('My bar chart !')
plt.show()
```
19
```import matplotlib.pyplot as plt
ages=[2,5,79,75,86,21,42,80,12,46,90,23,25,99,83,67]
range=(0,100)
bins=10
plt.hist(ages,bins,range,color='green',histtype='bar',rwidth=0.8)
plt.xlabel('age')
plt.ylabel('No. of people')
plt.title('My histogram')
plt.show()
```
20
```import matplotlib.pyplot as plt
import numpy as np
x=np.array([5,7,8,2,17,2,9,4,11,12,9,6])
y=np.array([99,86,87,111,86,103,87,94,78,77,85,86])
plt.scatter(x,y)
plt.show()
```
21
```import pandas as pd
df=pd.read_csv('/content/exams.csv')
print(df)
```
22
```import numpy as np
from sklearn.svm import SVC
x=np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y=np.array(['boy','boy','girl','girl'])
ml=SVC()
ml.fit(x,y)
result=ml.predict([[-1,-1]])
print(result)
```
23
```import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
ml=SVC()
ml.fit(x,y)
result=ml.predict([[6.3,3.3,6.0,2.5]])
if result==0:
  print("iris setosa")
elif result==1:
  print("iris vesicolor")
else:
  print("iris virginica")
```
24
```import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
ml=SVC()
ml.fit(x_train,y_train)
result=ml.predict(x_test)
print("Test input\n")
print(x_test)
print("Test output\n")
print(y_test)
print("predicted results\n")
print(result)
score=accuracy_score(result,y_test)
print(score)
```
25
```import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=pd.read_csv("/content/iris.csv")
iris.columns
```
26
```x=iris[['sepal.length','sepal.width','petal.length','petal.width']]
y=iris['variety']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
ml=SVC()
ml.fit(x,y)
result=ml.predict(x_test)
print(result)
```
27
```import numpy as np
arr1=np.array([1,2,3])
print("First array\n",arr1)
arr2=np.array([4,5,6])
print("Second array\n",arr2)
arr=np.concatenate((arr1,arr2))
print("concatenated array\n",arr)
newarr=np.array_split(arr,3)
print(newarr)
x=np.where(arr==4)
print("Found at:",x,"position")
```
28
```import numpy as np
arr=np.array([3,2,0,1])
print(np.sort(arr))
arr=np.array([41,42,43,44])
x=[True,False,True,False]
newarr=arr[x]
print(newarr)
```
29
```import numpy as np
arr=np.array([41,42,43,44])
filter_arr=[]
for element in arr:
  if element> 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr=arr[filter_arr]
print(filter_arr)
print(newarr)
```
30
```import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
img=np.array(Image.open('/moon.jpg'))
plt.figure(figsize=(4,6))
plt.imshow(img)
```
31
```#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```
32
```car_data=pd.read_csv('CarPrice_Assignment.csv')
```
33
```plt.figure(figsize=(18,8))
plt.subplot(1,3,1)
sns.countplot(x="symboling",data=car_data)
plt.title("symbolising Histogram")

plt.subplot(1,3,2)
sns.boxplot(x="symboling",y="price",data=car_data)
plt.title("symbolising Vs Price")

plt.subplot(1,3,3)
sns.countplot(x="symboling",hue="fueltype",data=car_data)
plt.title("Symboling Vs Fuel Type")
plt.show()
```
34
```plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.countplot(x="fueltype",data = car_data)
plt.title(" Fuel Type Histogram")

plt.subplot(1,2,2)
sns.countplot(x="carbody",data = car_data,order=car_data["carbody"].value_counts().index)
plt.title("Car Type Price")

plt.show()

