from   array  import array
import numpy  as np
import pandas as pd

import gzip
import os

class DataLoader():
    """"
    Base Class for DataLoaders

    Parameters
    ----------
        train_size   : {float, 'auto'}
                       Proportion of data belonging to the "training" dataset if float, otherwise the Child DataLoader defines it.

        random_state : {int, None}
                       Random seed for reproducible random results. If not set, a random_state is generated.

        path         : {str}
                       Path to directory containing the "datasets" sub-directory.
    """
    def __init__(self, train_size, random_state, path, **kwargs):

        assert (isinstance(train_size, (float)) and 0 < train_size < 1) or train_size == 'auto'
        assert isinstance(random_state, int) or random_state is None
        assert os.path.exists(path) if path else True

        self.train_size   = train_size
        self.random_state = int(np.random.uniform(0, 1000)) if random_state is None else random_state
        self.path         = path

        self.rng          = np.random.default_rng(random_state)

        if path and ('mnist' not in path):
            self.raw_data  = pd.read_csv(self.path)
            n              = len(self.raw_data)
            self.assign_idx(n, train_size)

        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def assign_idx(self, n, train_size):
        """ Randomly assigns indices as train or test """
        self.split     = int(n * train_size) + 1
        idx            = self.rng.permutation(n)
        self.train_idx = idx[:self.split]
        self.test_idx  = idx[self.split:]

    def get_idx(self, split, shuffle):
        """ Returns the indices of dataset """
        if split == 'train':
            idx = self.train_idx
        elif split == 'test':
            idx = self.test_idx
        elif split is None:
            idx = np.append(self.train_idx, self.test_idx)
        else:
            raise Exception()

        return self.rng.permutation(idx) if shuffle else idx

    def load(self, split = None, shuffle = False, **kwargs):
        """ Returns (X, [y]) """
        idx  = self.get_idx(split, shuffle)
        X, y = self._load(idx, **kwargs)
        return X if y is None else (X, y)

    @property
    def desc(self):
        desc = []
        for i, line in enumerate(self._desc.split('\n')):
            desc.append(line[8:] if i else line)
        print('\n'.join(desc))

class MNISTLoader(DataLoader):
    """
    MNIST DataLoader
    """
    def __init__(self, train_size = 'auto', random_state = None, path = '.'):
        url      = 'http://yann.lecun.com/exdb/mnist/'
        fullpath = os.path.join(path, 'datasets', 'classification', 'mnist')
        super().__init__(train_size, random_state, fullpath, url = url)

        X1, y1   = self._load_gz('train')
        X2, y2   = self._load_gz('t10k')

        n1, n2   = map(len, [X1, X2])

        self.raw_data = np.concatenate([X1, X2], axis = 0), np.concatenate([y1, y2], axis = 0)

        if train_size == 'auto':
            
            self.split     = int(n1 / (n1 + n2)) + 1
            self.train_idx = self.rng.permutation(n1)
            self.test_idx  = self.rng.permutation(n2) + n1

        else:

            self.split     = int(train_size * (n1 + n2)) + 1
            idx            = self.rng.permutation(n1 + n2)
            self.train_idx = idx[:self.split]
            self.test_idx  = idx[self.split:]

        target     = "{0, ..., 9}"
        self._desc = \
        f"""
        Handwritten Digit Image Classification.
        
        Feature (28, 28) in [0, 255].
        Target in {target}.

        By default, assumes a set train-test split.

        See {url} for more details.
        """.strip()


    def _load_gz(self, split):
        with gzip.open(os.path.join(self.path, f'{split}-images-idx3-ubyte.gz')) as gz:
            images = np.array(array('B', gz.read()).tolist()[16:]).reshape(-1, 28, 28)

        with gzip.open(os.path.join(self.path, f'{split}-labels-idx1-ubyte.gz')) as gz:
            labels = np.array(array('B', gz.read()).tolist()[8:])
        
        return images, labels

    def _load(self, idx):
        return self.raw_data[0][idx], self.raw_data[1][idx]


class NYCLoader(DataLoader):
    """
    NYC East River Bicycle Crossings DataLoader
    """
    def __init__(self, train_size = 0.8, random_state = None, path = '.', scenario = 'a'):
        url                 = 'https://www.kaggle.com/datasets/new-york-city/nyc-east-river-bicycle-crossings'
        fullpath            = os.path.join(path, 'datasets', 'regression', 'nyc-east-river-bicycle-counts.csv')
        super().__init__(train_size, random_state, fullpath, url = url)
        
        df                  = self.raw_data.copy()
        dt                  = pd.to_datetime(df['Day']).dt
        df['Precipitation'] = df['Precipitation'].apply(lambda val : float(val.split()[0]) if val != 'T' else 0)
        df                  = pd.concat([pd.get_dummies(dt.dayofweek), pd.get_dummies(dt.dayofyear), df.iloc[:,3:]], axis = 1)
        if scenario == 'b':
            columns = df.columns.copy()

            for i in range(37, len(columns)):
                df.insert(37, 'Previous ' + columns[36 - i], 0)

            df.iloc[1:,37:45] = df.iloc[:-1,45:].values

            df = df.loc[np.arange(210) % 30 != 0].copy()

        self.raw_data = df.copy()
        
        self.assign_idx(len(self.raw_data), train_size)

        self._desc = \
        f"""
        Bicycle Count Poisson Regression.
        
        Scenarios
        ---------
            a : (default)
                Feature (45) : day of week (7), day of year (35), temperature (2), precipitation (1).
                Target   (5) : count of bicycle at bridge location (4), total count (1).

                n = 210

            b : 
                Feature (53) : day of week (7), day of year (35) , previous temperature (2), previous precipitation (1), previous targets (5), temperature (2), precipitation (1).
                Target   (5) : count of bicycle at bridge location (4), total count (1).

                n = 203

        See {url} for more details.
        """.strip()

    def _load(self, idx):
        
        X = self.raw_data.values[idx,:-5]
        y = self.raw_data.values[idx,-5:]
        return X, y

class CandyLoader(DataLoader):
    """
    Candy Ranking DataLoader

    """
    def __init__(self, train_size = 0.8, random_state = None, path = '.'):
        url           = 'https://www.kaggle.com/datasets/fivethirtyeight/the-ultimate-halloween-candy-power-ranking'
        fullpath      = os.path.join(path, 'datasets', 'classification', 'candy-data.csv')
        super().__init__(train_size, random_state, fullpath, url = url)

        obs           = "{0, 1}"
        self._desc    = \
        f"""
        Candy Stat Percentage Regression

        Feature (9) in {obs}.
        Target (3) in [0, 1] (sugar, price, win).

        See {url} for more details.
        """.strip()

    def _load(self, idx):

        X    = self.raw_data.values[idx,1:-3]
        y    = self.raw_data.values[idx,-3:]
        
        return X, y

class BostonLoader(DataLoader):
    """
    Boston House Prices DataLoader
    """
    def __init__(self, train_size = 0.8, random_state = None, path = None):
        url = 'http://lib.stat.cmu.edu/datasets/boston'
        super().__init__(train_size, random_state, path, url = url)

        df  = pd.read_csv(url, sep = "\s+", skiprows = 22, header = None)

        X   = np.hstack([df.values[::2, :], df.values[1::2, :2]])
        y   = df.values[1::2, [2]]
        c   = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

        self.raw_data = pd.DataFrame(np.c_[X, y], columns = c)

        self.assign_idx(len(X), train_size)

        self._desc = \
        f"""
        Boston House Price Regression

        Feature (13) in R (non-negative).
        Target in R (non-negative).

        See {url} for more details.
        """
        
    def _load(self, idx):

        X = self.raw_data.values[idx,:-1]
        y = self.raw_data.values[idx,-1]

        return X, y
