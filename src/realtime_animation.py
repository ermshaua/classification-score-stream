import numpy as np
import daproli as dp
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()
sns.set_color_codes()


class ClaSPSegmetationStreamViewer:

    def __init__(self, ts_name, css, score="ClaSP Score", font_size=18, frame_rate=24):
        self.ts_name = ts_name
        self.css = css
        self.score = score
        self.font_size = font_size
        self.frame_rate = frame_rate

        self.init_animation()

    def init_animation(self):
        plt.ioff()

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': .05}, figsize=(20, 10))

        self.ax1.set_xlim((0, self.css.n_timepoints))
        self.ax1.set_ylim((0., 1.))

        self.ax2.set_xlim((0, self.css.n_timepoints))
        self.ax2.set_ylim((0., 1.))

        self.ax1.set_title(self.ts_name, fontsize=self.font_size)
        self.ax2.set_xlabel('split point  $s$', fontsize=self.font_size)
        self.ax2.set_ylabel(self.score, fontsize=self.font_size)

        for ax in (self.ax1, self.ax2):
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(self.font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(self.font_size)

        self.segment_lines = list()
        self.cp_lines = list()

        self.profile_line, = self.ax2.plot([], [], lw=2)
        self.profile_line.set_data([], [])

        self.frame_counter = 0


    def update_animation(self, ts):
        found_cps = self.css.change_points
        profile = self.css.profile

        ind, vals = [list() for _ in range(len(found_cps)+1)], [list() for _ in range(len(found_cps)+1)]

        not_ninf = np.logical_and(np.logical_not(np.isinf(ts)), np.logical_not(np.isnan(ts)))
        if ts[not_ninf].shape[0] != 0 and ts[not_ninf].min() != ts[not_ninf].max():
            extra = .05 * (ts[not_ninf].max() - ts[not_ninf].min())
            self.ax1.set_ylim((ts[not_ninf].min() - extra, ts[not_ninf].max() + extra))

        not_ninf = profile != -np.inf
        if profile[not_ninf].shape[0] != 0 and profile[not_ninf].min() != profile[not_ninf].max():
            extra = .05 * (profile[not_ninf].max() - profile[not_ninf].min())
            self.ax2.set_ylim((profile[not_ninf].min() - extra, profile[not_ninf].max() + extra))

        bucket_ptr = 0

        # set ts data
        for idx, val in enumerate(ts):
            global_idx = self.css.ingested - self.css.ts_stream_lag - ts.shape[0] + idx

            if global_idx in found_cps:
                bucket_ptr += 1

            ind[bucket_ptr].append(idx)
            vals[bucket_ptr].append(val)

        line_ptr = 0

        for line in self.segment_lines:
            line.set_data([], [])

        for idx, val in zip(ind, vals):
            if len(self.segment_lines) == line_ptr:
                line, = self.ax1.plot([], [], lw=2)
                self.segment_lines.append(line)

            self.segment_lines[line_ptr].set_data(idx, val)
            self.segment_lines[line_ptr].set_color(f'C{line_ptr}')
            line_ptr += 1

        # set profile data
        self.profile_line.set_data(np.arange(profile.shape[0]), profile)
        self.profile_line.set_color('black')

        line_ptr = 0
        rel_cp_counter = 0
        cp_counter = 0

        for line in self.cp_lines:
            line.set_data([], [])

        # set found_cps data
        for _ in self.css.change_points:
            if len(self.cp_lines) == line_ptr:
                line = self.ax1.axvline(lw=2)
                line.set_data([], [])
                self.cp_lines.append(line)

            if rel_cp_counter < len(found_cps):
                rel_cp = ts.shape[0] - self.frame_counter + found_cps[cp_counter]
                rel_cp_counter += 1
            else:
                rel_cp = -1

            self.cp_lines[line_ptr].remove()
            self.cp_lines[line_ptr] = self.ax2.axvline(x=rel_cp, linewidth=2, c='g', label=f'Predicted Change Point' if cp_counter == 0 else None)

            cp_counter += 1
            line_ptr += 1 # 2


    def update(self, timepoint):
        # log new data point
        self.frame_counter += 1

        # update ClaSP
        res = self.css.update(timepoint)

        if self.frame_counter <= self.css.n_prerun:
            prerun = self.css.prerun_ts
            ts = np.hstack((np.full(shape=self.css.n_timepoints-self.css.n_prerun, fill_value=np.nan), prerun))
        else:
            ts = self.css.ts_stream.time_series

        if self.frame_counter % self.frame_rate == 0:
            ts[np.isinf(ts)] = np.nan
            self.update_animation(ts)
            plt.pause(1e-15)

        return res

    @property
    def change_points(self):
        return self.css.change_points