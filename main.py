import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statistics

kfold_results_dt = pd.DataFrame(columns=["random state", "accuracy", "one off", "two off", "fail"])
kfold_results_rf = pd.DataFrame(columns=["random state", "accuracy", "one off", "two off", "fail"])
kfold_results_nb = pd.DataFrame(columns=["random state", "accuracy", "one off", "two off", "fail"])

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


def runner(k):
    # k-fold
    rs = 1
    for i in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=rs)
        # decision trees
        y_pred_dt = decision_tree_impl(X_train, y_train, X_test)
        accuracy_dt = metrics.accuracy_score(y_test, y_pred_dt)
        result_dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dt})
        save_results(kfold_results_dt, rs, result_dt, accuracy_dt)

        # random forest
        y_pred_rf = random_forest_impl(X_train, y_train, X_test)
        accuracy_rf = metrics.accuracy_score(y_test, y_pred_rf)
        result_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
        save_results(kfold_results_rf, rs, result_rf, accuracy_rf)

        # naive bayes
        y_pred_nb = random_forest_impl(X_train, y_train, X_test)
        accuracy_nb = metrics.accuracy_score(y_test, y_pred_nb)
        result_nb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
        save_results(kfold_results_nb, rs, result_nb, accuracy_nb)

        rs = rs + 13 * 2

    kfold_results_dt.to_csv("./results/kfold-results-dt.csv", index=False)
    kfold_results_rf.to_csv("./results/kfold-results-rf.csv", index=False)
    kfold_results_nb.to_csv("./results/kfold-results-nb.csv", index=False)
    print("standard deviation for decision trees: " + str(statistics.stdev(kfold_results_dt.loc[:, "accuracy"])))
    print("standard deviation for random forests: " + str(statistics.stdev(kfold_results_rf.loc[:, "accuracy"])))
    print("standard deviation for naive bayes: " + str(statistics.stdev(kfold_results_nb.loc[:, "accuracy"])))


def decision_tree_impl(X_train, y_train, X_test):
    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(splitter="random", random_state=0)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    # Predict the response for test dataset
    return clf.predict(X_test)


def random_forest_impl(X_train, y_train, X_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def naive_bayes_impl(X_train, y_train, X_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def save_results(result_set, rs, result, accuracy):
    result_list = result.values.tolist()
    size = len(result_list)

    one_off_count = 0
    two_off_count = 0
    fail_count = 0
    for i in range(size):
        row = result_list[i]
        diff = abs(row[0] - row[1])
        if diff == 0:
            continue
        elif diff == 1:
            one_off_count += 1
        elif diff == 2:
            two_off_count += 1
        else:
            fail_count += 1
        one_off_per = one_off_count * 100 / size
        two_off_per = two_off_count * 100 / size
        fail_per = fail_count * 100 / size

    result_set.loc[result_set.size] = [rs, accuracy, one_off_per, two_off_per, fail_per]


if __name__ == '__main__':
    runner(10)
