# imports
import warnings
import numpy as np
import pandas as pd
from scipy import stats
# Sklearn
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR, SVC
from sklearn.metrics import confusion_matrix
# Utils
from ML_utils.ml_utils_reg import tanimoto_from_dense
from ML_utils.utils import convert_to_class_actives
# Pytorch/ Skorch
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier, NeuralNetRegressor

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


class NeuralNetwork(nn.Module):

    def __init__(self, input_size: int = 2048, n_layers: int = 3, hidden_size: int = 128, dropout: float = 0,
                 output_size: int = 1, reg_class: str = None):
        super().__init__()

        self.reg_class = reg_class
        self.output_size = output_size
        self.l = nn.ModuleList()
        self.l.append(nn.Linear(input_size, hidden_size))
        self.l.append(nn.ReLU())

        for i in range(n_layers):
            self.l.append(nn.Linear(hidden_size, hidden_size))
            self.l.append(nn.ReLU())
            self.l.append(nn.Dropout(dropout))

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        for layer in self.l:
            x = layer(x)

        if self.reg_class == 'regression':
            out = self.out(x)

        elif self.reg_class == 'classification':
            out = torch.softmax(self.out(x), dim=-1)

        return out


class MLModel:
    def __init__(self, data, ml_algorithm, opt_metric=None, reg_class=None, class_type=None,
                 parameters='grid', cv_fold=5, random_seed=2002):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.class_type = class_type
        self.opt_metric = opt_metric
        self.reg_class = reg_class
        self.cv_fold = cv_fold
        self.seed = random_seed
        self.parameters = parameters
        self.data_encoder = self.data_encoder()
        self.h_parameters = self.hyperparameters()
        self.model, self.cv_results = self.cross_validation()
        self.best_params = self.optimal_parameters()
        self.model = self.final_model()

    def data_encoder(self):
        data_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(self.data.labels.reshape(-1, 1))
        return data_encoder
    def hyperparameters(self):
        if self.parameters == "grid":

            if self.reg_class == "regression":
                if self.ml_algorithm == "MR":
                    return {'strategy': ['median']
                            }
                elif self.ml_algorithm == "MeanR":
                    return {'strategy': ['mean']
                            }
                elif self.ml_algorithm == "SVR":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 10000],
                            }
                elif self.ml_algorithm == "RFR":
                    return {'n_estimators': [50, 100, 200],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_split': [2, 3, 5, 10],
                            'min_samples_leaf': [1, 2, 5, 10],
                            }
                elif self.ml_algorithm == "DNNR":
                    return {'module__n_layers': [2, 3],
                            'module__hidden_size': [128, 256, 512, 1024],
                            'module__dropout': [0.0],
                            'lr': [1e-4, 0.001, 0.01],
                            'max_epochs': [100, 200],
                            'batch_size': [16, 32, 64],
                            }

            elif self.reg_class == "classification" or self.reg_class == "classification-cw":
                if self.ml_algorithm == "SVC":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 10000],
                            }
                elif self.ml_algorithm == "RFC":
                    return {'n_estimators': [50, 100, 200],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_split': [2, 3, 5, 10],
                            'min_samples_leaf': [1, 2, 5, 10],
                            }
                elif self.ml_algorithm == "MC":
                    return {'strategy': ['prior']
                            }

    def cross_validation(self):

        if self.reg_class == "regression":
            opt_metric = self.opt_metric
            if self.ml_algorithm == "MR" or self.ml_algorithm == "MeanR":
                model = DummyRegressor()
            elif self.ml_algorithm == "SVR":
                model = SVR(kernel=tanimoto_from_dense)
            elif self.ml_algorithm == "RFR":
                model = RandomForestRegressor(random_state=self.seed)
            elif self.ml_algorithm == "DNNR":
                model = NeuralNetRegressor(module=NeuralNetwork,
                                           module__input_size=self.data.features.shape[1],
                                           module__output_size=1,
                                           module__reg_class=self.reg_class,
                                           criterion=nn.MSELoss,
                                           optimizer=torch.optim.Adam,
                                           max_epochs=100,
                                           lr=1e-3,
                                           train_split=False,
                                           verbose=0,
                                           device='cuda' if torch.cuda.is_available() else 'cpu',
                                           batch_size=16,
                                           )

        elif self.reg_class == "classification" or self.reg_class == "classification-cw":
            opt_metric = self.opt_metric
            if self.reg_class == "classification":
                if self.ml_algorithm == "SVC":
                    model = SVC(kernel=tanimoto_from_dense, random_state=self.seed, decision_function_shape='ovr',
                                probability=True, break_ties=True)
                elif self.ml_algorithm == "RFC":
                    model = RandomForestClassifier(random_state=self.seed)

                elif self.ml_algorithm == "DNNC":
                    model = NeuralNetClassifier(module=NeuralNetwork,
                                                module__input_size=self.data.features.shape[1],
                                                module__output_size=self.data_encoder.categories_[0].shape[0],
                                                module__reg_class='classification',
                                                criterion=nn.CrossEntropyLoss,
                                                optimizer=torch.optim.Adam,
                                                max_epochs=100,
                                                lr=1e-3,
                                                train_split=False,
                                                verbose=0,
                                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                                batch_size=16,
                                                )

            elif self.reg_class == "classification-cw":
                if self.ml_algorithm == "SVC":
                    model = SVC(kernel=tanimoto_from_dense, random_state=self.seed, decision_function_shape='ovr',
                                probability=True, class_weight='balanced', break_ties=True)
                elif self.ml_algorithm == "RFC":
                    model = RandomForestClassifier(random_state=self.seed, class_weight='balanced')

        cv_results = GridSearchCV(model,
                                  param_grid=self.h_parameters,
                                  cv=self.cv_fold,
                                  scoring=opt_metric,
                                  n_jobs=14, verbose=2, refit=False)

        if self.ml_algorithm == "DNNR":
            cv_results.fit(self.data.features.astype(np.float32), self.data.labels.astype(np.float32))

        elif self.ml_algorithm == "DNNC":
            data_encoder = self.data_encoder
            labels_enc = data_encoder.transform(self.data.labels.reshape(-1, 1))
            cv_results.fit(self.data.features.astype(np.float32), labels_enc.astype(np.float32))

        else:
            cv_results.fit(self.data.features, self.data.class_labels)

        return model, cv_results

    def optimal_parameters(self):
        best_params = self.cv_results.cv_results_['params'][self.cv_results.best_index_]
        return best_params

    def final_model(self):

        model = self.model.set_params(**self.best_params)

        if self.ml_algorithm == "DNNR" or self.ml_algorithm == "DNNC":
            return model.fit(self.data.features.astype(np.float32), self.data.labels.astype(np.float32).reshape(-1, 1))
        else:
            return model.fit(self.data.features, self.data.labels)


class Model_Evaluation:
    def __init__(self, model, data, model_id=None, model_loaded=None, model_type=None, reg_class=None,
                 active_target=None, potency_range=None, class_type=None, class_task=None,
                 min_pot=None, max_pot=None):

        self.reg_class = reg_class
        self.model_id = model_id
        self.model = model
        self.model_type = model_type
        self.data = data
        self.model_loaded = model_loaded
        self.active_target = active_target
        self.potency_range = potency_range
        self.class_type = class_type
        self.class_task = class_task
        self.min_pot, self.max_pot = min_pot, max_pot

        (self.labels, self.y_pred, self.predictions, self.labels_reg,
         self.y_prediction_reg, self.y_proba) = self.model_predict(data)

        self.pred_performance_class = self.prediction_performance_classification(data)

        if self.model_type == 'Global':
            self.pred_performance_class_global = self.prediction_performance_class_global()

    def model_predict(self, data):

        if self.reg_class == "regression":
            # fingerprints
            data_features = data.features
            # predictions regression
            if self.model_id == "DNNR":
                y_prediction_reg = self.model.predict(data_features.astype(np.float32)).flatten()
            else:
                y_prediction_reg = self.model.predict(data_features)
            # dummy proba
            y_proba = [None] * len(y_prediction_reg)
            # convert to class
            y_prediction, _ = convert_to_class_actives(y_prediction_reg, class_type=self.class_type,
                                                       min_pot=self.min_pot, max_pot=self.max_pot)
            # experimental potency
            labels_reg = self.data.labels

        elif self.reg_class == "classification" or self.reg_class == "classification-cw":
            # fingerprints
            data_features = data.features
            # predictions classification
            y_prediction = self.model.predict(data_features)
            # prediction uncalibrated probability
            y_proba = self.model.predict_proba(data_features)

            if self.class_type == 'active-inactive':
                y_prediction_reg = [None] * len(y_prediction)
                labels_reg = [None] * len(y_prediction)
            else:
                # prediction potency
                y_prediction_reg = [None] * len(y_prediction)
                # experimental potency
                labels_reg = [None] * len(y_prediction)

        # experimental class labels
        labels = self.data.class_labels
        # prepare prediction dataframe
        df_predictions = pd.DataFrame(list(zip(data.cid, labels_reg, y_prediction_reg, labels,
                                               y_prediction, y_proba, data.potency_class)),
                                      columns=["cid", "Experimental_reg", "Predicted_reg",
                                               "Experimental_class", "Predicted_class",
                                               "y_proba", "Potency_class"])

        df_predictions['Target ID'], df_predictions['Algorithm'] = self.data.target, self.model_id
        df_predictions['Class_type'], df_predictions['Class_task'] = self.class_type, self.class_task

        return labels, y_prediction, df_predictions, labels_reg, y_prediction_reg, y_proba

    def prediction_performance_classification(self, data) -> pd.DataFrame:

        labels = self.labels
        pred = self.y_pred
        y_proba = self.y_proba
        model_name = self.model_id

        if self.model_type == 'Global':
            target = 'Global'
        else:
            target = data.target[0]

        cnf_matrix = confusion_matrix(labels, pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - ((cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)) + (
                cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)) + (np.diag(cnf_matrix)))

        # calculate metrics binary classification
        if np.unique(labels).shape[0] == 2 or self.class_type == 'active-inactive':

            result_list = [{"MCC": metrics.matthews_corrcoef(labels, pred),
                            "F1": metrics.f1_score(labels, pred),
                            "AUC": metrics.roc_auc_score(labels, pred),
                            "BA": metrics.balanced_accuracy_score(labels, pred),
                            "BA weighted": metrics.balanced_accuracy_score(labels, pred, sample_weight=labels),
                            "Accuracy": metrics.accuracy_score(labels, pred),
                            "Precision": metrics.precision_score(labels, pred),
                            "Recall": metrics.recall_score(labels, pred),
                            "Average Precision": metrics.average_precision_score(labels, pred),
                            "Probability": y_proba,
                            "Dataset size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name,
                            "FP": FP,
                            "FN": FN,
                            "TP": TP,
                            "TN": TN,
                            }]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            # display(results)
            results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MCC",
                                                                      "F1",
                                                                      "AUC",
                                                                      "BA",
                                                                      "BA weighted",
                                                                      "Accuracy",
                                                                      "Precision", "Recall",
                                                                      "Average Precision",
                                                                      "Probability",
                                                                      "FP", "FN", "TP", "TN"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results

        else:
            # calculate metrics multi-class classification
            result_list = [{"MCC": metrics.matthews_corrcoef(labels, pred),
                            "BA": metrics.balanced_accuracy_score(labels, pred),
                            "BA weighted": metrics.balanced_accuracy_score(labels, pred, sample_weight=labels),
                            "F1 weighted": metrics.f1_score(labels, pred, average='weighted'),
                            "F1 macro": metrics.f1_score(labels, pred, average='macro'),
                            "Accuracy": metrics.accuracy_score(labels, pred),
                            "Precision macro": metrics.precision_score(labels, pred, average='macro'),
                            "Recall macro": metrics.recall_score(labels, pred, average='macro'),
                            "Precision micro": metrics.precision_score(labels, pred, average='micro'),
                            "Recall micro": metrics.recall_score(labels, pred, average='micro'),
                            "Probability": y_proba,
                            "Dataset size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name,
                            "FP": FP,
                            "FN": FN,
                            "TP": TP,
                            "TN": TN,
                            }]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            # display(results)
            results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MCC",
                                                                      "BA",
                                                                      "BA weighted",
                                                                      "F1 weighted",
                                                                      "F1 macro",
                                                                      "Accuracy",
                                                                      "Precision macro",
                                                                      "Recall macro",
                                                                      "Precision micro",
                                                                      "Recall micro",
                                                                      "Probability",
                                                                      "FP", "FN", "TP", "TN"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results

    def prediction_performance_regression(self, data) -> pd.DataFrame:

        labels = self.labels_reg
        pred = self.y_prediction_reg
        model_name = self.model_id

        if self.model_type == 'Global':
            target = 'Global'
        else:
            target = data.target[0]

        # calculate metrics
        mae = mean_absolute_error(labels, pred)
        mse = metrics.mean_squared_error(labels, pred)
        rmse = metrics.mean_squared_error(labels, pred, squared=False)
        r2 = metrics.r2_score(labels, pred)
        r = stats.pearsonr(labels, pred)[0]

        results_list = [{"MAE": mae,
                         "MSE": mse,
                         "RMSE": rmse,
                         "R2": r2,
                         "r": r,
                         "r2": r ** 2,
                         "Dataset size": len(labels),
                         "Target ID": target,
                         "Algorithm": model_name,
                         }]

        # Prepare result dataset
        results = pd.DataFrame(results_list)
        results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
        results.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2", "r", "r2"]],
                                                     names=["Value", "Metric"])
        results = results.stack().reset_index().set_index("Target ID")

        return results

    def prediction_performance_class_global(self) -> pd.DataFrame:

        result_list = []
        df_predictions = self.predictions

        for target in df_predictions['Target ID'].unique():
            # filter target
            df_predictions_target = df_predictions[df_predictions['Target ID'] == target]

            # df predictions/target
            labels = df_predictions_target.Experimental_class.values
            pred = df_predictions_target.Predicted_class.values
            y_proba = df_predictions_target.y_proba.values
            model_name = self.model_id

            # calculate confusion matrix
            cnf_matrix = confusion_matrix(labels, pred)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - ((cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)) + (
                    cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)) + (np.diag(cnf_matrix)))

            # calculate metrics multi-class classification
            results_tid = [{"MCC": metrics.matthews_corrcoef(labels, pred),
                            "BA": metrics.balanced_accuracy_score(labels, pred),
                            "BA weighted": metrics.balanced_accuracy_score(labels, pred, sample_weight=labels),
                            "F1 weighted": metrics.f1_score(labels, pred, average='weighted'),
                            "F1 macro": metrics.f1_score(labels, pred, average='macro'),
                            "Accuracy": metrics.accuracy_score(labels, pred),
                            "Precision macro": metrics.precision_score(labels, pred, average='macro'),
                            "Recall macro": metrics.recall_score(labels, pred, average='macro'),
                            "Precision micro": metrics.precision_score(labels, pred, average='micro'),
                            "Recall micro": metrics.recall_score(labels, pred, average='micro'),
                            "Probability": y_proba,
                            "Dataset size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name,
                            "FP": FP,
                            "FN": FN,
                            "TP": TP,
                            "TN": TN,
                            }]
            result_list.extend(results_tid)

        # Prepare result dataset
        results = pd.DataFrame(result_list)
        # display(results)
        results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
        results.columns = pd.MultiIndex.from_product([["Value"], ["MCC",
                                                                  "BA",
                                                                  "BA weighted",
                                                                  "F1 weighted",
                                                                  "F1 macro",
                                                                  "Accuracy",
                                                                  "Precision macro",
                                                                  "Recall macro",
                                                                  "Precision micro",
                                                                  "Recall micro",
                                                                  "Probability",
                                                                  "FP", "FN", "TP", "TN"]],
                                                     names=["Value", "Metric"])
        results = results.stack().reset_index().set_index("Target ID")

        return results
