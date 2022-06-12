import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# define column names
col_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
             'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# load dataset
# data = pd.read_csv("dataset/winequality-white.csv", sep=';', header=None, names=col_names)
data = pd.read_csv("dataset/winequality-red.csv", header=None, names=col_names)

X = data[feature_cols]  # Features
y = data.quality  # Target variable


def runner():
    calculate_feature_importance()

    # separation for validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_sub_train, X_validation, y_sub_train, y_validation = train_test_split(X_train, y_train, test_size=0.1)


def calculate_feature_importance():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    feature_importance = pd.DataFrame(clf.feature_importances_, index=col_names[0:-1],
                                      columns=['importance']).sort_values('importance', ascending=False)
    feature_importance.to_csv("./results/random-forest-feature-importance.csv", index=False)


if __name__ == '__main__':
    runner()
