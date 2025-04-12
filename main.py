from sklearn import datasets
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import metrics
from preprocess import preprocess_image

digits = datasets.load_digits()
X = digits.data
y = digits.target
print("le jeu de données : ",digits.data, "\n")
print("feature names : ",digits.feature_names,"\n")
print("target names : ",digits.target_names, "\n")
print("DESCRIPTION : \n", digits.DESCR)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=1)

print("X_train.shape : ", X_train.shape)
print("y_train.shape : ", y_train.shape)

model = SVC()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

exactitude = metrics.accuracy_score(y_test, y_predict)

print("accuracy : ", exactitude)

f_score = metrics.f1_score(y_test, y_predict, average="macro")

print("f1 score : ", f_score)

precision= metrics.precision_score(y_test, y_predict, average="macro") 

print("precision : ", precision)

rappel = metrics.recall_score(y_test, y_predict, average="macro")
print("rappel : ", rappel)


image_path = "./digit_images/8/8.png"
new_image_data = preprocess_image(image_path)

prediction = model.predict([new_image_data])

print("predicted digit : ", prediction[0])