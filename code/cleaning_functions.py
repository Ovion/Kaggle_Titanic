import pandas as pd
import re
import random

'''
Este script realiza las funciones de limpieza.
Con el fin de tenerlas separadas y que a la hora de entrenar nos pida si queremos realizar de nuevo la limpieza.
'''


def read_data(path_train='inputs/train.csv', path_test='inputs/test.csv'):
    '''
    Esta función recibe como parámetros los path de train y test y devuelve el data set
    por defecto:
        path_train='inputs/train.csv',
        path_test='inputs/test.csv'
    '''
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    return train, test


def courtesy(value):
    '''
    Esta función recibe como parámetro un string de un dataframe y su función es encontrar en dicho string la clase a la que pertenece esa persona.
    Devuelve dicha clase para poder asignarlo a una nueva columna
    '''

    court = {
        r"Mr\.": "Mr",
        r"Mr\s": "Mr",
        r"Mrs\.": "Mrs",
        r"Mrs\s": "Mrs",
        r"Ms\.": "Mrs",
        r"Miss\.": "Miss",
        r"Master\.": "Master",
        r"Dr\.": "Dr",
        r"Rev\.": "Rev",
        r"Col\.": "Col",
        r"Capt\.": "Capt",
        r"Major\.": "Major"
    }
    for k, v in court.items():
        if re.search(k, value):
            return v
    else:
        return "Other"


def create_court(df):
    '''
    Esta función recibe un df que contenga una columna llamada "Name" y crea la columna "Court" basado en al función 'courtesy'
    '''
    df['Court'] = df.Name.apply(lambda x: courtesy(str(x)))
    return df


def fillna_age(df):
    '''
    Esta función rellena los NaN encontrados en la columna "Age" del df, y los rellena en base a la columna "Court"
    utiliza los percentiles .25, .5 y .75
    '''
    lst_court = list(df.Court.unique())

    for court in lst_court:
        descr = df.loc[df.Court == court].Age.describe()

        for indx, _ in df.loc[(df.Court == court) & (df.Age.isna())].iterrows():
            rnd = random.randint(0, 100)
            if rnd < 25:
                df.loc[indx, 'Age'] = random.randint(
                    int(descr['25%']-descr['std']), int(descr['25%']+descr['std']))
            elif 25 <= rnd <= 50:
                df.loc[indx, 'Age'] = random.randint(
                    int(descr['50%']-descr['std']), int(descr['50%']+descr['std']))
            else:
                df.loc[indx, 'Age'] = random.randint(
                    int(descr['75%']-descr['std']), int(descr['75%']+descr['std']))
    return df


def last_clean_dummy(df,):
    '''
    Esta función recibe un df para aplicar los últimos cambios y hacer un dummies de ['Sex', 'Embarked', 'Court'].
    '''
    df.Embarked.fillna('S', inplace=True)
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    dummy = pd.get_dummies(data=df, columns=['Sex', 'Embarked', 'Court'])
    return dummy
