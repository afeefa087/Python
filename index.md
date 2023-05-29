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
