from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def cross_validation(target, features):

  zeror = DummyClassifier(strategy="most_frequent")
  logistic = LogisticRegression(max_iter=10000)
  knn = KNeighborsClassifier()
  j48 = DecisionTreeClassifier()
  rf = RandomForestClassifier()

  # Realiza a validação cruzada com 10 partições
  scores_zeror = cross_val_score(zeror, features, target, cv=10)
  scores_logistic = cross_val_score(logistic, features, target, cv=10)
  scores_knn = cross_val_score(knn, features, target, cv=10)
  scores_j48 = cross_val_score(j48, features, target, cv=10)
  scores_rf = cross_val_score(rf, features, target, cv=10)

  return {
      "DummyClassifier": scores_zeror.mean(),
      "LogisticRegression": scores_logistic.mean(),
      "KNeighborsClassifier": scores_knn.mean(),
      "DecisionTreeClassifier": scores_j48.mean(),
      "RandomForestClassifier": scores_rf.mean()
  }