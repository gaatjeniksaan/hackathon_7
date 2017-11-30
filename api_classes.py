from sklearn.base import BaseEstimator, TransformerMixin

class OneHotDataFramer(BaseEstimator, TransformerMixin):
    """One-hot encodes a single column of a pd.DataFrame.
    
    Parameters
    ----------

    col : str
        Column to select (if None assumes that input is a
        DataFrame consisting of a single column).
    """
    
    def __init__(self, col=None):
        self.col_ = col
        self.binarizer = LabelBinarizer()
        
    def fit(self, X, y=None):
        df = X[self.col_]
        self.binarizer.fit(X[self.col_])
        return self

    def get_feature_names(self):
        return self.binarizer.classes
        
    def transform(self, X, y=None):
        return self.binarizer.transform(X[self.col_])
