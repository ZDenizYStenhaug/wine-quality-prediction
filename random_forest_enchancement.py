import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import statistics


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # separation for validation set
    X_sub_train, X_validation, y_sub_train, y_validation = train_test_split(X_train, y_train, test_size=0.1,
                                                                            random_state=0)
    optimized_params = fine_tune_parameters(X_sub_train, y_sub_train)

    compare(optimized_params)

    calculate_feature_importance(X_train, y_train)


def random_forest_impl(X_train, y_train, X_test, optimized_params):
    clf = RandomForestClassifier(random_state=0,
                                 n_estimators=optimized_params["n_estimators"],
                                 min_samples_split=optimized_params["min_samples_split"],
                                 min_samples_leaf=optimized_params["min_samples_leaf"],
                                 max_features=optimized_params["max_features"],
                                 max_depth=optimized_params["max_depth"],
                                 criterion=optimized_params["criterion"])
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def compare(optimized_params):
    accuracy_list = []
    # 10-fold
    rs = 1
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=rs)
        y_pred_rf = random_forest_impl(X_train, y_train, X_test, optimized_params)
        accuracy_list.append(metrics.accuracy_score(y_test, y_pred_rf))
        rs = rs + 13 * 2

    print(accuracy_list)
    optimized_avg = statistics.mean(accuracy_list)
    optimized_std = statistics.stdev(accuracy_list)

    # read from summary csv
    initial = pd.read_csv('results/kfold-summary.csv')
    cols = ["", "avg accuracy", "st deviation"]
    summary = [["initial", initial['avg accuracy'].values[1], initial['st deviation'].values[1]],
               ["optimized", optimized_avg, optimized_std]]
    df = pd.DataFrame(summary, columns=cols)
    df.to_csv("./results/optimization-summary.csv", index=False)


def fine_tune_parameters(X_sub_train, y_sub_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ["sqrt", "log2"]
    # The function to measure the quality of a split.
    criterion = ["gini", "entropy", "log_loss"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # The minimum number of samples required to split an internal node
    min_samples_split = [2, 3, 5, 7, 10]
    # The minimum number of samples required to be at a leaf node
    min_samples_leaf = [1, 2, 3]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'criterion': criterion}

    clf = RandomForestClassifier()

    optimized_rf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=5, cv=3, verbose=2,
                                      random_state=0, n_jobs=-1)
    # Fit the random search model
    optimized_rf.fit(X_sub_train, y_sub_train)
    optimized_parameters = pd.DataFrame(optimized_rf.best_params_.values(), index=optimized_rf.best_params_.keys(),
                                        columns=["values"])
    optimized_parameters.to_csv("./results/optimized-parameters.csv")
    return optimized_rf.best_params_


def calculate_feature_importance(X_train, y_train):

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    feature_importance = pd.DataFrame(clf.feature_importances_, index=col_names[0:-1],
                                      columns=['importance']).sort_values('importance', ascending=False)
    feature_importance.to_csv("./results/random-forest-feature-importance.csv")


if __name__ == '__main__':
    runner()
