import numpy as np

from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self, priors=None, n_dependencies=0, init_current_state=False):
        self._priors = priors
        self._dependencies = None
        self._n_dependencies = n_dependencies
        self._dependencies_keys = []
        if isinstance(init_current_state, bool):
            self.samples = np.array([])
        else:
            self.samples = np.array([init_current_state])

    def _check_for_dependencies(self):
        if self._dependencies is None:
            raise ValueError("DEPENDENCIES NOT SUPPLIED FOR MODEL!")
        if len(self._dependencies) != self._n_dependencies:
            raise ValueError("INCORRECT NUMBER OF DEPENDENCIES SUPPLIED")

    @property
    def priors(self):
        return self._priors

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, dependencies):
        if not isinstance(dependencies, dict):
            raise ValueError("DEPENDENCIES MUST BE DICTIONARY TYPE")
        self._check_dependencies_keys(dependencies)
        self._dependencies = dependencies

    def _check_dependencies_keys(self, dependencies):
        missing_keys = []
        for expected_key in self._dependencies_keys:
            if expected_key not in dependencies.keys():
                missing_keys.append(expected_key)
        if len(missing_keys) > 0:
            raise ValueError(f"MISSING EXPECTED KEYS FOR MODEL: {missing_keys}")

    @abstractmethod
    def sample_full_conditional(self):
        raise NotImplementedError

    @abstractmethod
    def get_dependencies(self):
        raise NotImplementedError

    @abstractmethod
    def get_priors(self):
        raise NotImplementedError

    @abstractmethod
    def get_sufficient_statistics(self):
        raise NotImplementedError

    @property
    def current_state(self):
        return self.samples[-1]
