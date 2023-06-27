from pyflink.datastream.functions import ProcessWindowFunction

from src.clazz.profile import binary_f1_score
from src.clazz.segmentation import ClaSS
from src.clazz.window_size import suss


class ClaSSProcessWindowFunction(ProcessWindowFunction):

    def __init__(self, n_timepoints=10_000, n_prerun=None, window_size=suss,
                 k_neighbours=3, score=binary_f1_score, jump=5, p_value=1e-50,
                 sample_size=1_000, similarity="pearson"):
        self.n_timepoints = n_timepoints
        self.n_prerun = n_prerun
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = score
        self.jump = jump
        self.p_value = p_value
        self.sample_size = sample_size
        self.similarity = similarity

        self.stream = None
        self.last_cp = None

    def open(self, runtime_context):
        self.stream = ClaSS(
            n_timepoints=self.n_timepoints,
            n_prerun=self.n_prerun,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            score=self.score,
            jump=self.jump,
            p_value=self.p_value,
            sample_size=self.sample_size,
            similarity=self.similarity
        )

    def process(self, key, context, elements):
        for timepoint in elements:
            self.stream.update(timepoint)

            if len(self.stream.change_points) > 0 and self.stream.change_points[-1] != self.last_cp:
                self.last_cp = self.stream.change_points[-1]
                yield self.last_cp

    def close(self):
        self.stream = None
