import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print('Preparing data...')
data = pd.read_csv('inputs/train_clean.csv')
test = pd.read_csv('inputs/test_clean.csv')

X = data.drop(['Survived', 'PassengerId'], axis=1)
X_test = test.drop(['PassengerId'], axis=1)
y = data.Survived

print('Doing some magic...')
stdsca = StandardScaler()
X = pd.DataFrame(stdsca.fit_transform(X))
X_test = pd.DataFrame(stdsca.fit_transform(X_test))

rfc = RandomForestClassifier(
    n_jobs=-1, n_estimators=500, min_samples_leaf=4, min_samples_split=5)
rfc.fit(X, y)
y_pred = rfc.predict(X_test)

submit = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': y_pred
})

print('Saving at outputs/')
submit.to_csv('outputs/submit_silva.csv', index=False)
