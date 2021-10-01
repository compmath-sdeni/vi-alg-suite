from methods.IterativeAlgorithm import IterativeAlgorithm
from problems.viproblem import VIProblem


class IterGradTypeMethod(IterativeAlgorithm):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters: int = 0, max_iters: int = 5000, hr_name: str = None):
        self.problem: VIProblem
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters, hr_name=hr_name)
