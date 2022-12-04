from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import metrics 
from sklearn import model_selection
from lazypredict.Supervised import LazyClassifier

class FeatureSelection:
    def __init__(self, X, Y, model=None, verbose=True):
        self.X = X
        self.Y = Y
        self.verbose = verbose
        if model is None:
            self.models = [self.variance_threshold,
                        #    self.sequential,
                        #    self.svm_rfe,
                           self.univariate,
                           self.L1,
                           self.tree_based]

    def variance_threshold(self):
        if self.verbose:
            print('==== MODEL: VARIANCE THRESHOLD ====')
        from sklearn.feature_selection import VarianceThreshold

        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_new = sel.fit_transform(self.X)
        self.lazy_predict(X_new, self.Y)
        return [[X_new.shape[1], self.predict(X_new, self.Y)]]

    def sequential(self, direction='forward',k_list=None):
        if self.verbose:
            print('==== MODEL: SEQUENTIAL {} ===='.format(direction.upper()))
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SequentialFeatureSelector

        if k_list is None:
            k_list = range(1, self.X.shape[1], int(self.X.shape[1] * 0.2))

        # Build RF classifier to use in feature selection
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        results = []
        for k in k_list:
            print('K =', k, end=', ')
            sfs = SequentialFeatureSelector(clf, n_features_to_select=k, direction=direction)
            sfs.fit(self.X, self.Y)
            X_new = sfs.transform(self.X)
            self.lazy_predict(X_new, self.Y)
            results.append([X_new.shape[1], self.predict(X_new, self.Y)])
        
        return results

    def svm_rfe(self):
        # if self.verbose:
        #     print('==== MODEL: SVM RFE ====')
        # estimator = SVR(kernel='linear')
        # selector = RFE(estimator, n_features_to_select=2, step=1)
        # selector = selector.fit(X_train, Y_train)
        # selector.support_
        # selector.ranking_
        pass

    def univariate(self, k_list=None):
        if self.verbose:
            print('==== MODEL: UNIVARIATE ====')
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        if k_list is None:
            k_list = range(1, self.X.shape[1], int(self.X.shape[1] * 0.1))

        results = []
        for k in k_list:
            print('K =', k, end=', ')
            X_new = SelectKBest(chi2, k=k).fit_transform(self.X, self.Y)
            self.lazy_predict(X_new, self.Y)
            results.append([X_new.shape[1], self.predict(X_new, self.Y)])
        return results

    def L1(self):
        if self.verbose:
            print('==== MODEL: L1-BASED ====')
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(self.X, self.Y)
        model = SelectFromModel(lsvc, prefit=True)

        X_new = model.transform(self.X)
        self.lazy_predict(X_new, self.Y)
        return [[X_new.shape[1], self.predict(X_new, self.Y)]]

    def tree_based(self):
        if self.verbose:
            print('==== MODEL: TREE-BASED ====')
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(self.X, self.Y)

        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(self.X)
        self.lazy_predict(X_new, self.Y)
        return [[X_new.shape[1], self.predict(X_new, self.Y)]]

    def predict(self, X, Y):
        X_train, y_train, X_test, y_test = self.train_test_split(X, Y)
        # creating a RF classifier
        clf = RandomForestClassifier(n_estimators = 100) 
        
        # Training the model on the training dataset
        # fit function is used to train the model using the training sets as parameters
        clf.fit(X_train, y_train)
        
        # performing predictions on the test dataset
        y_pred = clf.predict(X_test)
        
        # using metrics module for accuracy calculation
        print("ACCURACY OF THE MODEL: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
        return metrics.accuracy_score(y_test, y_pred)
    
    def lazy_predict(self, X, Y):
        X_train, y_train, X_test, y_test = self.train_test_split(X, Y)
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        print(models)

    def train_test_split(self, X, Y):
        Y = Y.astype('float')
        X = X.astype('float')
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)
        # X_train = X_train.apply(pd.to_numeric) 
        # X_test = X_test.apply(pd.to_numeric)
        
        return X_train, Y_train, X_test, Y_test