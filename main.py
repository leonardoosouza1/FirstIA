import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

file = pd.read_csv('data/wine.csv')

file.head()

file['style'] = file['style'].replace('red', 0)
file['style'] = file['style'].replace('white', 1)

y = file['style']
x = file.drop('style', axis=1)

x_learing, x_test, y_learning, y_test = train_test_split(x, y, test_size=0.3)

modelo = ExtraTreesClassifier()
modelo.fit(x_learing, y_learning)

result = modelo.score(x_test, y_test)
print("Resultado: ", result)
