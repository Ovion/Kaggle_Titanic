import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier

data = pd.read_csv('inputs/train_clean.csv')

X = data.drop(['Survived', 'PassengerId'], axis=1)
y = data.Survived

stdsca = StandardScaler()
X = pd.DataFrame(stdsca.fit_transform(X))

models = {
    "GBC": GradientBoostingClassifier(n_estimators=500, min_samples_leaf=4, min_samples_split=5),
    "RFC": RandomForestClassifier(n_jobs=-1, n_estimators=500, min_samples_leaf=4, min_samples_split=5),
}

for mod_name, model in models.items():
    print(f'Training model: {mod_name}')
    # Entrenando para cada modelo del diccionario
    lst_accu = []
    lst_prec = []
    for i in range(20):
        # Mi propio Crossvalidation
        print(f'Iteration number {i}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = round(accuracy_score(y_test, y_pred), 4)
        lst_accu.append(accuracy)

        precision = round(precision_score(
            y_test, y_pred, average='weighted'), 4)
        lst_prec.append(precision)

    print("Saving data at 'output/records.txt'...")
    # Tengo un log de todo lo que voy haciendo
    accu_mean = round(sum(lst_accu)/len(lst_accu), 2)
    prec_mean = round(sum(lst_prec)/len(lst_prec), 2)
    params = model.get_params
    with open('outputs/records.txt', "a+") as file:
        file.write(
            f'''Model: {mod_name}\t Accuracy: {accu_mean}%\t Precision: {prec_mean}%
            \n\tParams: {params} \n\n'''
        )
    print(f'Model {mod_name} analyzed')
