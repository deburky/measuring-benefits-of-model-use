import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class Classifiers(object):
    """
    Classifier object for fitting, storing, and comparing multiple model outputs.
    """

    def __init__(self, classifier_list, meta_model=None):

        self.classifiers = classifier_list
        self.classifier_names = [est[0] for est in self.classifiers]

        # List to store pipeline objects for classifiers
        self.pipelines = []

        # Set default integration model if not specified
        if meta_model is None:
            self.meta_model = LogisticRegression(fit_intercept=True,
                                                 solver='newton-cg',
                                                 penalty='none',
                                                 max_iter=1000,
                                                 random_state=42)
        else:
            self.meta_model = meta_model

    def create_pipelines(self, mapper):

        for i, classifier in enumerate(self.classifiers):
            self.pipelines.append(Pipeline([
                ('featurize', mapper),
                ('classifier', classifier[2])
                ]))

#         self.pipelines.append(pipeline)

    def train(self, X, y):
        """
        Training of meta model
        """

        X_meta = np.zeros((X.shape[0], len(self.classifiers)))
        
        for i, clf in enumerate(self.classifiers):
            mask = clf[1] # column names to include
            model = clf[2].fit(X[mask], y.values.ravel()) # train model
            proba = np.clip(model.predict_proba(X[mask])[:,1], 0.01, 0.99)   
            X_meta[:, i] = np.log(proba / (1 - proba)) # logits
    
        self.meta_model.fit(X_meta, y.values.ravel())
        
        trained_pipeline = Pipeline([
            ('classifier', self.meta_model)
        ])
        
        self.pipelines.append(trained_pipeline)
        
        return self.meta_model
    
    def predict_proba(self, meta_model, X):
        """
        Prediction of meta model
        """
        
        X_meta = np.zeros((X.shape[0], len(self.classifiers)))
        
        # make predictions with each classifier
        for t, clf in enumerate(self.classifiers):
            mask = clf[1] # column names to include
            proba = np.clip(clf[2].predict_proba(X[mask])[:,1], 0.01, 0.99)
            X_meta[:, t] = np.log(proba / (1 - proba)) # logits

        return meta_model.predict_proba(X_meta)