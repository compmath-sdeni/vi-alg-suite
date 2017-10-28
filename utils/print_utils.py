import numpy as np


def scalarToString(v: float, digits: int = 0) -> str:
    return "{0:.8f}".format(v) if digits == 0 else ("{0:." + str(digits) + "f}").format(v)


def vectorToString(x: np.ndarray) -> str:
    if len(x.shape) == 1:
        return str([("{0}" if isinstance(v, str) else "{0:.7f}").format(v) for v in
                    (x if len(x) <= 5 else [x[0], x[1], '...', x[-2], x[-1]])])
    else:
        return "\n".join([v if isinstance(v, str) else vectorToString(v) for v in
                          (x if len(x) <= 5 else [x[0], x[1], '...', x[-2], x[-1]])])
