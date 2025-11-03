# Extragradient algorithms for VI test suite #

Software system to apply algorithms for variational inequalities for different problems and compare/analyze 
their behaviour. 

## Summary ##
Python-based solution where different combinations of algorithms can be used to solve a problem formulated as 
a variational inequality (VI) in the form:

find vector x from closed convex feasible set C such, that

![<Ax, y-x> >= 0 for all y in C>](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%5Cleft%3CAx,y-x%5Cright%3E%5Cge%200%5C;%5Cforall%20y%5Cin%20C%20)

After the run, the system provides a detailed log of the process and allows to get several types of convergency graphs. 
Horizontal axis can be either number of iterations or time. 
Vertical axis can be either step size (norm of the difference between previous and next approximations), 
GAP (if there is a way to compute it for the problem), or 
distance to exact solution (if known).

## Brief ##
The system is based on the following components: 
* Problem definition
* Algorithms
* Feasible set definition
* Coordination scripts

### Problem definition ###
Basically, to be solved by an extragradient type of algorithm, 
the problem must allow to compute operator value (gradient for optimization type of problems) and projection 
to the feasible set.

In the current implementation, the suite also expects a problem class to provide gap function (if known), 
get distance from the current approximation to the real solution (if known) and have method to return dictionary
of name-value pairs (again, based on the current approximation to the solution) for some problem-specific metrics - so
behaviour of the algorithms can be analyzed in more details using iterations log.

Problem classes make a hierarchy, with `problems/Problem` class as a base class, VIProblem is a bit more specific 
descendant, and all specific problems are descendants of `VIProblem`. For samples, you can look at 
`problems/FuncNDMin` or `problems/PseudoMonotoneOperOne` classes. 

Here are the main methods for a problem class:
* `def A(self, x: np.ndarray) -> np.ndarray` - calculates operator value at point x
* `def F(self, x: np.ndarray) -> float:` - calculates function value at point x (goal 
function for optimization, or gap function, if applicable)
* `def Project(self, x: np.array) -> np.array:` - projects point x to the feasible set
* `def saveToDir(self, *, path_to_save: str = None):` - saves problem data to the directory
* `def loadFromDir(self, *, path_to_load: str = None):` - loads problem data from the directory
* `def bregmanProject(self, x:np.ndarray, a: np.ndarray) -> np.ndarray:` - projects point x to a using Bregman projection, 
applicable to selected problems and algorithms.

Problem creation logic is often quite complex, so it may be convenient to use dedicated functions
to generate test cases - examples of such an approach can be found in `problems/testcases` - e.g. 
`problems/testcases/pseudo_mono_3.py` or `problems/testcases/blood_delivery/blood_delivery_test_three.py`.

To set specific parameters for algorithms for the testcase, object of type `AlgorithmParams` is used - 
where some parameters are set initially, and all needed changes are made in the test case generator code 
(see mentioned examples).

### Feasible set (constraint) definition ###
Standard way for problem class to calculate the projection is to delegate it to the feasible set class.
Feasible set classes also make a simple hierarchy, with `constraints/ConvexSetConstraints` base class.
All other specific sets are descendants of `ConvexSetConstraints`. For examples, you can look at `constraints/L1Ball` or 
`constraints/RnPlus` classes. Also, there is a special class, `constraints/ConvexSetsIntersection`, which 
allows to combine several sets into one.

Here are the main methods for a feasible set class:
* `def project(self, x: np.ndarray) -> np.ndarray` - projects point x to the set
* `def isIn(self, x: np.ndarray) -> bool` - checks if point x is in the set
* `def getSomeInteriorPoint(self) -> np.ndarray` - returns some arbitrary interior point of the set
* `def saveToDir(self, path: str)` - saves set data to the directory
 

### Algorithms ###
Algorithms class hierarchy is based on the following classes:
* `methods/IterativeAlgorithm` - base class for all iterative algorithms
* `methods/IterGradTypeMethod` - a bit more specific descendant class, which is a base for all iterative gradient-type algorithms

All specific methods classes, such as `methods/Tseng` for Tseng's algorithm, are descendants of `IterGradTypeMethod`.

Here are the main methods for an algorithm class:
* `def __iter__(self)` - part of python iterator interface, initialization. It's implemented in base class, and can be 
partially overridden in descendants - usually, you will do some extra logic here and call `return super().__iter__()` in the end. 
* `def __next__(self) -> dict` - part of python iterator interface, iteration step. It's implemented in base class as 
checking stop condition with `isStopConditionMet()` method and calling `doStep()` and `doPostStep()`.
So, in many cases it will be enough to override `doStep()` and `doPostStep()` methods in descendants.
The `__next__` method also responsible for logging the process with measuring time and adding records to in-memory log, 
* which can be used later and saved by the solver. 
* `def doStep(self)` - the main method, which does the actual step of the algorithm - for sure should be implemented 
in descendants.
* `def doPostStep(self)` - does some post-step logic, if needed for the algorithm. Can save some specific history data, 
recalculate extra parameters etc.
* `def isStopConditionMet(self)` - checks stop conditions, which can be different for different algorithms.
There are also some auxiliary methods, for saving history etc. - see, for example, 
`methods/tseng_adaptive.py` and `methods/tseng.py` files.

* The important feature of algorithm classes is the history accumulation logic - it's base is implemented in 
base class, see `setHistoryData` function, and also related calls in __iter__ and __next__ methods. It's supposed to be 
extended in descendants, if needed. The idea is that each algorithm can have (or not) some specific history data, which can be 
saved to the log and used later for analysis. Base history date includes current iteration time, current solution 
approximation, current step size and problem-specific metrics (see `problems/VIProblem` class, and especially 
GetExtraIndicators function).

### Solver ###

Solver is a convenience script to run algorithms for selected problems, plot and save graphs etc. 
In the current version, it's responsible for:
* Creating problem and feasible set objects
* Creating algorithm objects
* Running algorithms
* Saving results and history
* Plotting graphs

It uses auxiliary classes `BasicAlgoTests` and `AlgStatGrapher`, to print standard information about the run and draw 
corresponding graphs.

At the moment, the project includes three variants of solver scripts. The first is `main.py`, which can be used for 
conducting our numerical experiments, if it's more convenient for you to manipulate parameters from code, than to use 
command line or gui. The second is `run_algs_command.py`, which is a command-line wrapper for the same logic. And the 
last one is `run_algs_lib.py` with `AlgsRunner` class, which is intended to be used from another python code.

As a usage example, see plotly dash application in `webviews` subfolder, which contains editor for a blood supply 
chain problem and uses `AlgsRunner` to run algorithms and plot graphs 
(see `solve_problem_click` callback in `net-editor.py`). This application is planned to be extended 
to allow plugging in problems, algorithms and feasible sets.