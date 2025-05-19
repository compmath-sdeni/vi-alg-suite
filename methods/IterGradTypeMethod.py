from enum import Enum, unique

from methods.IterativeAlgorithm import IterativeAlgorithm
from methods.algorithm_params import StopCondition
from problems.viproblem import VIProblem


@unique
class ProjectionType(Enum):
    EUCLID = 1
    BREGMAN = 2


class IterGradTypeMethod(IterativeAlgorithm):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters: int = 0, max_iters: int = 5000, hr_name: str = None,
                 projection_type: ProjectionType = ProjectionType.EUCLID,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE, save_history: bool = True):
        self.problem: VIProblem
        self.projection_type: ProjectionType = projection_type
        self.averaged_result = None

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, stop_condition=stop_condition, save_history=save_history
                         )
