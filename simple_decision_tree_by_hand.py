# only for petal length
def simple_decision_tree(sample):
    if sample[2] <= 2.5:
        return "Setosa"
    elif sample[2] <= 5.0:
        return "Versicolor"
    else:
        return "Virginica"

# Sample data (sepal_len, sepal_wid, petal_len, petal_wid)
sample1 = [5.1, 3.5, 1.4, 0.2]
sample2 = [6.5, 2.8, 4.6, 1.5]
sample3 = [7.2, 3.0, 6.0, 1.8]

print(simple_decision_tree(sample1))
print(simple_decision_tree(sample2))
print(simple_decision_tree(sample3))
