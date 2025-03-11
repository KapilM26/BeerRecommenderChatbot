class IsDarkLabelEncoder:
    def __init__(self):
        self.mapping = {"light": 0, "dark": 1}

    def transform(self, X):
        return [self.mapping[val] for val in X]

    def single_transform(self, X):
        return self.mapping[X]

    def fit_transform(self, X):
        if type(X) == str:
            return self.single_transform(X)
        else:
            return self.transform(X)


class CountryLabelEncoder:
    def __init__(self):
        self.mapping = {"US": 0, "GB": 1, "PL": 2, "CA": 3}

    def transform(self, X):
        return [self.mapping.get(val, 4) for val in X]

    def single_transform(self, X):
        return self.mapping[X]

    def fit_transform(self, X):
        if type(X) == str:
            return self.single_transform(X)
        else:
            return self.transform(X)


class AvailabilityLabelEncoder:
    def __init__(self):
        self.mapping = {
            "Year-round": 0,
            "Winter": 1,
            "Spring": 2,
            "Summer": 3,
            "Fall": 4,
        }

    def transform(self, X):
        return [self.mapping[val] for val in X]

    def single_transform(self, X):
        return self.mapping[X]

    def fit_transform(self, X):
        if type(X) == str:
            return self.single_transform(X)
        else:
            return self.transform(X)


class FlavorLabelEncoder:
    def __init__(self):
        self.mapping = {"bitter": 0, "malty": 1, "fruity": 2}

    def transform(self, X):
        return [self.mapping[val] for val in X]

    def single_transform(self, X):
        return self.mapping[X]

    def fit_transform(self, X):
        if type(X) == str:
            return self.single_transform(X)
        else:
            return self.transform(X)


class AlcoholLabelEncoder:
    def __init__(self):
        self.mapping = {"Low": 0, "Medium": 1, "High": 2}

    def transform(self, X):
        return [self.mapping[val] for val in X]

    def single_transform(self, X):
        return self.mapping[X]

    def fit_transform(self, X):
        if type(X) == str:
            return self.single_transform(X)
        else:
            return self.transform(X)
