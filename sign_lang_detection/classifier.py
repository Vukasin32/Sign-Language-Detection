import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dict_ = pickle.load(open("./data.pickle", 'rb'))

data = np.array(dict_['data'])
labels = np.array(dict_['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'{accuracy_score(y_pred, y_test)*100}% accuracy!')

with open("model.pickle", "wb") as f:
    pickle.dump({'model': model}, f)
