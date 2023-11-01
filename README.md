# Raising the ClaSS of Streaming Time Series Segmentation

This is the supporting website for the paper <a href="https://arxiv.org/abs/2310.20431">"Raising the ClaSS of Streaming Time Series Segmentation"</a>. It contains the used source codes, the data sets, raw results, and analysis notebooks. It reflects the state of the paper for reproducibility and is purposely not further updated.

Ubiquitous sensors today emit high frequency streams of numerical measurements that reflect properties of human, animal, industrial, commercial, and natural processes. Shifts in such processes, e.g. caused by external events or internal state changes, manifest as changes in the recorded signals. The task of streaming time series segmentation (STSS) is to partition the stream into consecutive variable-sized segments that correspond to states of the observed processes or entities. The partition operation itself must in performance be able to cope with the input frequency of the signals. We introduce ClaSS, a novel, efficient, and highly accurate algorithm for STSS. ClaSS assesses the homogeneity of potential partitions using self-supervised time series classification and applies statistical tests to detect significant change points (CPs). In our experimental evaluation using two large benchmarks and six real-world data archives, we found ClaSS to be significantly more precise than eight state-of-the-art competitors. Its space and time complexity is independent of segment sizes and linear only in the sliding window size. We also provide ClaSS as a window operator with an average throughput of 538 data points per second for the Apache Flink streaming engine.

https://github.com/ermshaua/classification-score-stream/blob/main/videos/student_commute.mp4

## Benchmark Results

We have evaluated ClaSS and eight competitors on 107 benchmark and 485 data archive time series from experimental studies. The following table summarises the average Covering performance (higher is better) and the corresponding wins / ties. More details are in the paper. The raw measurements are <a target="_blank" href="https://github.com/ermshaua/classification-score-stream/blob/main/experiments">here</a> and analysis Jupyter notebooks are <a target="_blank" href="https://github.com/ermshaua/classification-score-stream/blob/main/notebooks/comparative_analysis/">here</a>.

| Segmentation Algorithm | Average (in %) | Std. Dev. (in %) | Wins & Ties (in %) |
|------------------------|---|--------------|--------------------
| ClaSS                  | 81.2 / 51.5 | 19.0 / 17.1  | 72.9 / 46.8        |
| ChangeFinder           | 47.3 / 42.3 | 23.5 / 19.7  | 11.2 / 19.6        |
| FLOSS                  | 52.1 / 35.6 | 22.7 / 13.0  | 11.2 / 9.3         |
| Window                 | 46.1 / 29.1 | 24.7 / 27.7  | 11.2 / 13.4        |
| DDM                    | 53.5 / 26.2 | 16.9 / 24.5  | 9.3 / 8.5          |
| BOCD                   | 48.1 / -  | 19.0 / -     | 7.5 / -            |
| ADWIN                  | 38.3 / 26.2  | 20.6 / 20.5  | 3.7 / 5.2          |
| HDDM                   | 36.5 / 24.6  | 24.8 / 18.5  | 4.7 / 4.3          |
| NEWMA                  | 43.4 / 21.5  | 20.6 / 26.2  | 8.4 / 9.3          |

## Organisation

This repository is structured in the following way: 

- `benchmark` contains the source codes used for running the paper experiments.
- `datasets` consists of the TSSB benchmark data sets.
- `experiments` contains the raw measurement results for ClaSS and the competitors. 
- `figures` includes the paper plots, generated by the Jupyter notebooks.
- `videos` includes the paper videos, generated by the animation code.
- `notebooks` consists of Jupyter notebooks, used to download data sets and analyse results.
- `src` contains the sources codes for ClaSS, the competitors and utility methods.

## Installation

You can download this repository (by clicking the download button in the upper right corner). As this repository is a supporting website and not an updated library, we do not recommend to install it! Extract or adapt code snippets of interest. We are currently working on integrating ClaSS as a part of the maintained and updated <a href="https://github.com/ermshaua/claspy" target="_blank">claspy</a> library.

## Citation

This paper is currently under review. After publishing, you can find the citation request here. You can find a preprint <a href="https://arxiv.org/abs/2310.20431">here</a>.

## Resources

The sources codes for the competitors in the benchmark evaluation come from multiple authors and projects. We list here the resources we used (and adapted) for our experiments:
- Evaluation Metrics (https://github.com/alan-turing-institute/TCPDBench)
- Change Finder (https://github.com/nel215/change_finder)
- FLOSS (https://stumpy.readthedocs.io/)
- Window (https://centre-borelli.github.io/ruptures-docs/)
- BOCD (https://github.com/y-bar/bocd)
- DDM, ADWIN & HDDM (https://scikit-multiflow.readthedocs.io/)
- NEWMA (https://github.com/lightonai/newma)