class Multi_1:
    def __init__(self, W1, W2, W3):
        self.W1 = int(W1)
        self.W2 = int(W2)
        self.W3 = int(W3)

    def compute(self, S, P):
        ret = S + P
        return ret + self.W1,ret+self.W2,ret+self.W3

class Multi_2:
    def compute(self, *kwargs):
        return sum(kwargs[0:-1]),sum(kwargs[1:])
