import numpy as np
from sklearn.svm import SVC
X = np.array([
    [180, 80],   
    [160, 55],   
    [170, 65],   
    [155, 50],   
    [175, 70]    
])
y = np.array([1, 0, 1, 0, 1])
model = SVC(kernel='linear')
model.fit(X, y)
w = model.coef_[0]      
b = model.intercept_[0]  
print("w1 =", w[0])
print("w2 =", w[1])
print("b  =", b)
new_sample = np.array([[175, 70]])
f_x = np.dot(w, new_sample[0]) + b
print("f(x) =", f_x)
prediction = model.predict(new_sample)
print("Predicted class:", prediction[0], "(Athlete)" if prediction[0] == 1 else "(Non-athlete)")
