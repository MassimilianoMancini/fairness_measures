import pandas as pd
from fairness_measures_api import fairness_measures_api
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


def cal(file, subset, algorithm, g0, g1):
    data = pd.read_csv(file, sep=';', header='infer')
    X = pd.get_dummies(data.drop('G3', axis=1))
    le = LabelEncoder()
    y = le.fit_transform(data['G3'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = algorithm
    model.fit(X_train, y_train)
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)

    X_test['r'] = predictions_test
    X_test['G3'] = data.iloc[X_test.index]['G3']
    X_test['y_hat'] = (X_test['r'] >= 10).astype('int')
    X_test['y'] = (X_test['G3'] >= 10).astype('int')

    X_train['r'] = predictions_train
    X_train['G3'] = data.iloc[X_train.index]['G3']
    X_train['y_hat'] = (X_train['r'] >= 10).astype('int')
    X_train['y'] = (X_train['G3'] >= 10).astype('int')

    d = eval(subset)

    f = fairness_measures_api(d, 'r', 10, 'y', 'y_hat', g0, g1)
    print ("==============================================")
    print (f'Algorithm is {model.__class__.__name__}')
    print (f'File is {file}')
    print (f'Subset is {subset}')
    print (f'Group 0 (positive) is {g0}')
    print (f'Group 1 (negative) is {g1}')
    print ("==============================================")

    print (f'{'True statistical parity':.<30} {f.true_statistical_parity():>6,.3f}')
    print (f'{'Model accuracy':.<30} {f.model_accuracy():>6,.3f}')
    print (f'{'Statistical parity':.<30} {f.statistical_parity():>6,.3f}')
    print (f'{'Total accuracy':.<30} {f.total_accuracy():>6,.3f}')
    print (f'{'Calibration':.<30} {f.calibration():>6,.3f}')
    print (f'{'Overall Fairness Index':.<30} {f.ofi():>6,.3f}')
    print ("==============================================")
    print ("==============================================")
    print()


# Main
files      = ['student-mat.csv', 'student-por.csv']
groups     = [('sex_F', 'sex_M'), ('romantic_no', 'romantic_yes')]
subsets    = ['X_test', 'X_train']
algorithms = [XGBClassifier(), KNeighborsClassifier(), HistGradientBoostingClassifier()]


for f in files:
    for (g0, g1) in groups:
        for s in subsets:
            for a in algorithms:
                cal(f, s, a, g0, g1)