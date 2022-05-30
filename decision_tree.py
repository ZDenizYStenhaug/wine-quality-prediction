import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

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
    rs = 1

    for i in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=rs)
        y_pred = decision_tree_impl(X_train, y_train, X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        print_results(result)

        rs = rs + 13 * 2



def decision_tree_impl(X_train, y_train, X_test):


    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(splitter="random", random_state=0)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    # visualize_tree(clf, feature_cols) - graph too large.

    # Predict the response for test dataset
    return clf.predict(X_test)


def print_results(result):
    result_list = result.values.tolist()

    accurate_count = 0
    one_off_count = 0
    two_off_count = 0
    fail_count = 0
    for i in range(len(result_list)):
        row = result_list[i]
        diff = abs(row[0] - row[1])
        if diff == 0:
            accurate_count += 1
        elif diff == 1:
            one_off_count += 1
        elif diff == 2:
            two_off_count += 1
        else:
            fail_count += 1
    print("Size of the test set:", len(result_list))
    print("accurate prediction:", accurate_count)
    print("predictions that were one off:", one_off_count)
    print("predictions that were two off:", two_off_count)
    print("predictions that failed:", fail_count)


def visualize_tree(clf, feature_cols):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, round=True,
                    special_characters=True, feature_names=feature_cols,
                    class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('wine_quality.png')
    Image(graph.create_png())


if __name__ == '__main__':
    runner(10)
