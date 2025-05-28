import numpy as np
from scipy import stats
from scipy.stats import rv_continuous

class MixtureModel(stats.rv_continuous):
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
            
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf

        

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


class Dirac(rv_continuous):
    def __init__(self, constant_value):
        super().__init__()
        self.constant_value = constant_value

    def _pdf(self, x):
        # Return 1 for the constant value and 0 otherwise
        return np.where(x == self.constant_value, 1, 0)

    def rvs(self, size=1, **kwargs):
        if isinstance(size, int):
            size = (size,)
        out = np.full(size, self.constant_value)
        return out


uniform = stats.uniform(loc=0, scale = 1)
uniform_t999 = stats.uniform(loc=0, scale = 0.999)
uniform_t001 = stats.uniform(loc=0.001, scale = 0.999)
uniform_low = stats.uniform(loc=0, scale = 0.025)
low_t_heavy = MixtureModel([
                              stats.uniform(loc=0, scale = 0.025),
                              stats.uniform(loc=0, scale = 1)],
                             weights = [0.25, 0.75])
dirac_1 = Dirac(1)
dirac_p5 = Dirac(0.5)
dirac_p4 = Dirac(0.4)