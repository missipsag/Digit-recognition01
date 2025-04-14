from sklearn import datasets
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import metrics
from preprocess import preprocess_image
from canvas import Paint
from PIL import ImageGrab



digits = datasets.load_digits()
X = digits.data
y = digits.target

# afficher le jeu de donnée
print("le jeu de données : ",digits.data, "\n")
# afficher les caractéristique du dataset
print("feature names : ",digits.feature_names,"\n")
# afficher les but à atteindre, en l'occurence les chiffre de 0 à 9
print("target names : ",digits.target_names, "\n")
# afficher la description du set 
print("DESCRIPTION : \n", digits.DESCR)

# deviser le dataset en deux partie, une partie pour le test et l'autre pour entrainer le model
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.4, random_state=0)

print("X_train.shape : ", X_train.shape)
print("y_train.shape : ", y_train.shape)

# SVC (Support Vector Classifier) est une implémentation de l'algorithme SVM.
# Il est utilisé dans les tâches de classication.
model = SVC(probability=True)

# on entraine le model sur le jeu de donnée
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

Paint()

new_image_data = preprocess_image("screenshot.png", invert=True)
print("new image data \n", new_image_data)

prediction = model.predict([new_image_data])

print("predicted digit : ", prediction[0])