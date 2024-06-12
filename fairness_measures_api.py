class fairness_measures_api:
    def __init__(self, d, r, y, h, g0, g1):
        self.d = d
        self.r = r
        self.y = y
        self.h = h
        self.g0 = g0
        self.g1 = g1

    def main():
        pass

    def true_statistical_parity(self):
        total_g0 = len(self.d[self.d[self.g0]])
        total_g1 = len(self.d[self.d[self.g1]])
        success_g0 = len(self.d[self.d[self.g0] & self.d[self.y]])
        success_g1 = len(self.d[self.d[self.g1] & self.d[self.y]])
        return success_g0/total_g0 - success_g1/total_g1
    
    def statistical_parity(self):
        total_g0 = len(self.d[self.d[self.g0]])
        total_g1 = len(self.d[self.d[self.g1]])
        success_g0 = len(self.d[self.d[self.g0] & self.d[self.h]])
        success_g1 = len(self.d[self.d[self.g1] & self.d[self.h]])
        return success_g0/total_g0 - success_g1/total_g1
    
    def total_accuracy(self):
        total_g0 = len(self.d[self.d[self.g0]])
        total_g1 = len(self.d[self.d[self.g1]])
        tp_g0 = len(self.d[self.d[self.g0] & self.d[self.y] & self.d[self.h]])
        tp_g1 = len(self.d[self.d[self.g1] & self.d[self.y] & self.d[self.h]])
        tn_g0 = len(self.d[self.d[self.g0] & ~self.d[self.y] & ~self.d[self.h]])
        tn_g1 = len(self.d[self.d[self.g1] & ~self.d[self.y] & ~self.d[self.h]])
        return ((tp_g0 + tn_g0)/total_g0) - ((tp_g1 + tn_g1)/total_g1) 
    
    def calibration(self):
        c = 0
        rs = self.d[self.r].unique()

        for single_r in rs:
            subset = self.d[self.d[self.r] == single_r]
            total_g0 = len(subset[subset[self.g0]])
            total_g1 = len(subset[subset[self.g1]])
            success_g0 = len(subset[subset[self.g0] & subset[self.y]])
            success_g1 = len(subset[subset[self.g1] & subset[self.y]])

            g0 = 0 if total_g0 == 0 else success_g0/total_g0
            g1 = 0 if total_g1 == 0 else success_g1/total_g1

            w = len(subset)/len(self.d)

            c += (g0 - g1) * w
        return c
    
    def ofi(self, w_sp = 1/3, w_ta = 1/3, w_ca = 1/3):
        return w_sp * self.statistical_parity() + w_ta * self.total_accuracy() + w_ca * self.calibration()
    
    if __name__ == "__main__":
        main()