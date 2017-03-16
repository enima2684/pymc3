from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from .utils import get_transformed


class NormalizingFlow:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def init(self, rvs):
        """Initialize transform parameters.

        If a random variable in :code:`rv` is a local random variable,
        the 1st dimension of the tensor is the size of mini-batches.
        
        rvs : list of random variables
            List of random variables.
        """
        pass
    
    @abstractmethod
    def trans(self, zs):
        """Transform samples of random variables."""
        pass
    
    @abstractmethod
    def ldj(self, zs):
        """Returns logdet of Jacobian."""
        pass
    
    @abstractmethod
    def get_params(self):
        """Returns transform parameters."""
        pass


def apply_normalizing_flows(zs, rvs, normalizing_flows, return_elbo):
    zs_new = OrderedDict([(rv.name, z) for rv, z in zip(rvs, zs)])
    elbo = 0

    for nf in normalizing_flows:
        zs_org = [zs_new[get_transformed(rv).name] for rv in nf[0]]
        zs_trn = nf[1].trans(*zs_org)
        zs_new.update(
            OrderedDict(
                [(rv.name, z_trn) for rv, z_trn in zip(nf[0], zs_trn)]
            )
        )
        if return_elbo:
            elbo += nf[1].ldj(*zs_org)

    if return_elbo:
        return list(zs_new.values()), elbo
    else:
        return list(zs_new.values())
