class BetaCalculator:
    def __init__(self, beta, max_iters):
        self.beta = beta
        self.max_iters = max_iters

    def calculate(self, iter):
        if iter < self.max_iters:
            return self.beta * ((self.max_iters - iter) / self.max_iters)
        else:
            return 0
