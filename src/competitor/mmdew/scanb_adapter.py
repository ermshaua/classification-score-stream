import numpy as np

from detectors import RegionalDriftDetector, DriftDetector

import onlinecp.algos as algos
import onlinecp.utils.feature_functions as feat


class ScanB(DriftDetector):
    def __init__(self, window_size, num_windows,thresholding_quantile, forget_factor):
        """
        window_size: int,
            size of time windows.
        nbr_windows: int,
            number of time windows to use in ScanB.      
        """
        self.window_size = window_size
        self.num_windows = num_windows
        self.thresholding_quantile = thresholding_quantile
        self.forget_factor = forget_factor
        self.detected_cp = False
        


        #detector.apply_to_data(X)
        super(ScanB, self).__init__()

    def name(self) -> str:
        return "ScanB" 

    def parameter_str(self) -> str:
        return r"$wsize = {}, numw = {}, thresholding_quantile={}, forget_factor={}$".format(self.window_size, self.num_windows, self.thresholding_quantile, self.forget_factor)

    def pre_train(self, data):
        big_Lambda, small_lambda = algos.select_optimal_parameters(self.window_size)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        d = data.shape[1]
        W, sigmasq = feat.generate_frequencies(m, d, data=data, choice_sigma="median")
        self.detector = algos.ScanB(data[0], kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(sigmasq)), window_size=self.window_size, nbr_windows=self.num_windows, adapt_forget_factor=self.forget_factor,thresholding_quantile=self.thresholding_quantile)
    
    

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

 
