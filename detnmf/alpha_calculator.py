from collections import deque


class AlphaCalculator:
    def __init__(self, alpha_scale, nmf_iters, max_iters):
        self.alpha_scale = alpha_scale
        self.nmf_iters = nmf_iters
        self.max_iters = max_iters
        self.recent = deque()
        self.final_alpha = None

    # allow use in a new set of iterations
    def reset(self):
        self.recent = deque()
        self.final_alpha = None

    def calculate(self, WTW, iter):
        if iter < self.nmf_iters:
            return 0
        if iter < self.max_iters:
            alpha = WTW.mean() * self.alpha_scale
            self.recent.append(alpha)
            if len(self.recent) > 30:
                self.recent.popleft()
            return alpha
        else:
            if self.final_alpha is None:
                self.final_alpha = sum(self.recent) / len(self.recent)
                if abs(self.final_alpha - self.recent[-1] ) > 0.01:
                    print("Alpha Did Not Converge.  Choosing Alpha=", self.final_alpha)
                    print("Can try using smaller alpha_scale to reduce oscillation, increasing max_iters, or change number of components")
            return self.final_alpha
