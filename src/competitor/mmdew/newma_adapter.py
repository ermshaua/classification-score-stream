import numpy as np

from detectors import RegionalDriftDetector, DriftDetector

import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat


class NewMA(DriftDetector):
    def __init__(self, window_size, thresholding_quantile, forget_factor):
        """
        window_size: int,
            size of time windows.
        nbr_windows: int,
            number of time windows to use in ScanB.      
        """

        self.window_size = window_size
        self.detected_cp = False
        self.thresholding_quantile=thresholding_quantile
        self.forget_factor = forget_factor
        super(NewMA, self).__init__()

    def name(self) -> str:
        return "NewMA" 

    def parameter_str(self) -> str:
        return r"$wsize={}, forget_factor={}, thresholding_quantile={}$".format(self.window_size, self.forget_factor, self.thresholding_quantile)

    def pre_train(self, data):
        big_Lambda, small_lambda = algos.select_optimal_parameters(self.window_size)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        d = data.shape[1]
        W, sigmasq = feat.generate_frequencies(m, d, data=data, choice_sigma="median")

        def feat_func(x):
            return feat.fourier_feat(x, W)
        self.detector = algos.NEWMA(data[0], forget_factor=self.forget_factor, feat_func=feat_func, adapt_forget_factor=self.forget_factor, thresholding_quantile=self.thresholding_quantile)
    
    

    def add_element(self, input_value):
        """
        Add the new element and also perform change detection
        :param input_value: The new observation
        :return:
        """

        self.detected_cp = self.detector.update(input_value[0])
        return

        self.element_count+=1
        self.detected_cp = False
        prev_cps = len(self.detector.get_changepoints())
        self.detector.insert(input_value[0])
        if len(self.detector.get_changepoints()) > prev_cps:
            self.delay = self.element_count - self.detector.get_changepoints()[-1]
            self.detected_cp = True
#            print("Detected")

    def detected_change(self):
        return self.detected_cp
    
    def metric(self):
        return 0

 
