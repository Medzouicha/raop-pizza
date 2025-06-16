import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from utils.seed import set_seed

set_seed(42)  # reproductible
X, y = make_classification(n_samples=500, n_features=20, random_state=42)

with mlflow.start_run(run_name="baseline_logreg"):
    clf = LogisticRegression(max_iter=500).fit(X, y)
    acc = clf.score(X, y)

    mlflow.log_param("model", "LogReg")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")  # sauvegarde du modèle
