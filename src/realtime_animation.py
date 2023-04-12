import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation

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

        ind, vals = [list() for _ in range(len(found_cps) + 1)], [list() for _ in range(len(found_cps) + 1)]

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
            self.cp_lines[line_ptr] = self.ax2.axvline(x=rel_cp, linewidth=2, c='g',
                                                       label=f'Predicted Change Point' if cp_counter == 0 else None)

            cp_counter += 1
            line_ptr += 1  # 2

    def update(self, timepoint):
        # log new data point
        self.frame_counter += 1

        # update ClaSP
        res = self.css.update(timepoint)

        if self.frame_counter <= self.css.n_prerun:
            prerun = self.css.prerun_ts
            ts = np.hstack((np.full(shape=self.css.n_timepoints - self.css.n_prerun, fill_value=np.nan), prerun))
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


class ClaSSAnimator:

    def __init__(self, ts_name, ts, css, true_cps=None, seg_names=None, file_path=None, score="ClaSP Score",
                 font_size=18, frame_rate=100):
        self.ts_name = ts_name
        self.ts = ts
        self.css = css
        self.true_cps = true_cps
        self.seg_names = seg_names
        self.file_path = file_path
        self.score = score
        self.font_size = font_size
        self.frame_rate = frame_rate

    def _calc_class(self):
        self.clasp_min, self.clasp_max = 1, 0
        self.windows = []

        for idx, timepoint in enumerate(self.ts):
            self.css.update(timepoint)

            if idx <= self.css.n_prerun:
                prerun = self.css.prerun_ts
                seq = np.hstack((np.full(shape=self.css.n_timepoints - self.css.n_prerun, fill_value=np.nan), prerun))
                profile = np.full(shape=seq.shape[0], fill_value=-np.inf, dtype=np.float64)
            else:
                seq = self.css.ts_stream.time_series
                profile = self.css.profile

            if idx % self.frame_rate == 0:
                ninf_profile = profile[np.logical_not(np.isneginf(profile))]
                if ninf_profile.shape[0] > 0: self.clasp_min = min(self.clasp_min, np.min(ninf_profile))
                self.clasp_max = max(self.clasp_max, np.max(profile))

                self.windows.append((idx, seq, profile, np.asarray(self.css.change_points)))

    def _init_animation(self):
        plt.clf()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': .05}, figsize=(20, 10))

        self.ax1.set_xlim((0, self.css.n_timepoints))
        self.ax1.set_ylim((self.ts.min(), self.ts.max()))

        self.ax2.set_xlim((0, self.css.n_timepoints))
        self.ax2.set_ylim((self.clasp_min, self.clasp_max))

        self.ax1.set_title(self.ts_name, fontsize=self.font_size)
        self.ax2.set_xlabel('split point  $s$', fontsize=self.font_size)
        self.ax2.set_ylabel(self.score, fontsize=self.font_size)

        for ax in (self.ax1, self.ax2):
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(self.font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(self.font_size)

    def __init_lines(self):
        self.lines = list()

        if self.true_cps is None:
            self.true_cps = np.array([], dtype=np.int64)

        # add lines for segments
        for _ in np.arange(len(self.true_cps) + 1):
            line, = self.ax1.plot([], [], lw=2)
            self.lines.append(line)

        # add lines for true cps
        for _ in np.arange(len(self.true_cps)):
            line = self.ax1.axvline(lw=2)
            self.lines.append(line)

            line = self.ax2.axvline(lw=2)
            self.lines.append(line)

        # add line for profile
        line, = self.ax2.plot([], [], lw=2)
        self.lines.append(line)

        # add lines for found cps
        for _ in np.arange(len(self.css.change_points)):
            line = self.ax1.axvline(lw=2)
            self.lines.append(line)

            line = self.ax2.axvline(lw=2)
            self.lines.append(line)

    def _init_func_animation(self):
        for line in self.lines: line.set_data([], [])
        return self.lines

    def _update_func_animation(self, frame):
        ts_idx, seq, clasp, found_cps = frame

        ind, vals = [list() for _ in range(len(self.true_cps) + 1)], [list() for _ in range(len(self.true_cps) + 1)]
        valid_cps = []
        bucket_ptr = 0
        line_ptr = 0

        # set ts data
        for idx, val in enumerate(self.ts[:ts_idx]):
            rel_idx = seq.shape[0] - ts_idx + idx

            if rel_idx < 0 or rel_idx > seq.shape[0]:
                continue

            if idx in self.true_cps:
                valid_cps.append(idx)
                bucket_ptr += 1

            ind[bucket_ptr].append(rel_idx)
            vals[bucket_ptr].append(val)

        colors = [f'C{idx}' for idx in range(len(ind))]

        valid_seg_names = []

        # set segment names
        if self.seg_names is not None:
            if len(valid_cps) > 0:
                for idx, cp in enumerate(self.true_cps):
                    if cp in valid_cps:
                        if cp == valid_cps[0]:
                            valid_seg_names.append(self.seg_names[idx])
                        valid_seg_names.append(self.seg_names[idx + 1])
            else:
                for idx, true_cp in enumerate(self.true_cps):
                    if len(ind[0]) == 0:
                        break

                    if ind[0][0] - (seq.shape[0] - ts_idx) < true_cp:
                        valid_seg_names.append(self.seg_names[idx])
                        break

            cmap = plt.get_cmap("tab20")
            seg_colors = cmap.colors
            seg_colors = {a: seg_colors[idx] for idx, a in enumerate(sorted(set(self.seg_names)))}

            for idx, seg_name in enumerate(valid_seg_names):
                colors[idx] = seg_colors[seg_name]

        for idx, vals in zip(ind, vals):
            self.lines[line_ptr].set_data(idx, vals)
            self.lines[line_ptr].set_color(colors[line_ptr])
            line_ptr += 1

        cp_counter = 0

        # set true cp data
        for cp in self.true_cps:
            rel_cp = seq.shape[0] - ts_idx + cp

            self.lines[line_ptr].remove()
            self.lines[line_ptr] = self.ax1.axvline(x=rel_cp, linewidth=2, c='r',
                                                    label=f'True Change Point' if cp_counter == 0 else None)

            self.lines[line_ptr + 1].remove()
            self.lines[line_ptr + 1] = self.ax2.axvline(x=rel_cp, linewidth=2, c='r',
                                                        label=f'True Change Point' if cp_counter == 0 else None)

            cp_counter += 1
            line_ptr += 2

        self.lines[line_ptr].set_data(np.arange(clasp.shape[0]), clasp)
        self.lines[line_ptr].set_color('black')
        line_ptr += 1

        rel_cp_counter = 0
        cp_counter = 0

        # set found cp data
        for _ in self.css.change_points:
            if rel_cp_counter < len(found_cps):
                rel_cp = seq.shape[0] - ts_idx + found_cps[cp_counter]
                rel_cp_counter += 1
            else:
                rel_cp = -1

            self.lines[line_ptr].remove()
            self.lines[line_ptr] = self.ax1.axvline(x=rel_cp, linewidth=2, c='g',
                                                    label=f'Predicted Change Point' if cp_counter == 0 else None)

            self.lines[line_ptr + 1].remove()
            self.lines[line_ptr + 1] = self.ax2.axvline(x=rel_cp, linewidth=2, c='g',
                                                        label=f'Predicted Change Point' if cp_counter == 0 else None)

            cp_counter += 1
            line_ptr += 2

        return self.lines + [self.ax1.legend(prop={'size': self.font_size}, loc=1)]

    def run_animation(self):
        self._calc_class()
        self._init_animation()
        self.__init_lines()

        self.animation = FuncAnimation(
            self.fig,
            self._update_func_animation,
            frames=self.windows,
            init_func=self._init_func_animation,
            interval=100,
            blit=True
        )

        if self.file_path is not None:
            self.animation.save(self.file_path)
