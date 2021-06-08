from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

for i in range(101):
    print(f'Index: {i}')
    if i == 50:
        break
else:
    print("Not found!")