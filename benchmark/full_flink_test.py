import os
import resource
import shutil
import sys
import time

sys.path.insert(0, "../")

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic

from src.clazz.flink import ClaSSProcessWindowFunction

import numpy as np

np.random.seed(1379)

from benchmark.utils import evaluate_candidate


def evaluate_flink_class(name, w, cps, ts, **seg_kwargs):
    if "n_timepoints" in seg_kwargs:
        n_prerun = min(seg_kwargs["n_timepoints"], ts.shape[0])
    else:
        n_prerun = min(10_000, ts.shape[0])

    if "window_size" in seg_kwargs and seg_kwargs["window_size"] == "predefined":
        seg_kwargs["window_size"] = w

    # Set up the environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_stream_time_characteristic(TimeCharacteristic.ProcessingTime)
    env.set_parallelism(1)

    # Read the input data stream
    input_stream = env.from_collection(ts.tolist(), type_info=Types.FLOAT())
    window_function = ClaSSProcessWindowFunction(n_prerun=n_prerun, **seg_kwargs)

    # Apply the ClaSSProcessWindowFunction
    output_stream = input_stream \
        .key_by(lambda x: 0) \
        .count_window(min(10_000, len(ts))) \
        .process(window_function, output_type=Types.INT())

    # Write the output data stream
    output_stream.print()

    # Before execution
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    runtime = time.process_time()

    # Execute the job
    env.execute("ClaSS Flink Test")

    # After execution
    runtime = time.process_time() - runtime
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory
    throughput = len(ts) / runtime

    # f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    # covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    print(f"{name}: Throughput: {throughput}")
    return name, runtime, throughput, memory


def evaluate_flink_class_dataset(dataset_name, exp_path, n_jobs, verbose):
    name = f"flink_{dataset_name}"

    if os.path.exists(exp_path + name):
        shutil.rmtree(exp_path + name)

    os.mkdir(exp_path + name)

    candidate_name, eval_func = "FlinkClaSS", evaluate_flink_class

    df = evaluate_candidate(
        candidate_name,
        dataset_name,
        eval_func=eval_func,
        n_jobs=n_jobs,
        columns=["dataset", "runtime", "throughput", "memory"],
        verbose=verbose,
    )

    df.to_csv(f"{exp_path}{name}/{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_flink_class_dataset("UTSA", exp_path, n_jobs, verbose)
    evaluate_flink_class_dataset("TSSB", exp_path, n_jobs, verbose)

    evaluate_flink_class_dataset("PAMAP", exp_path, n_jobs, verbose)
    evaluate_flink_class_dataset("mHealth", exp_path, n_jobs, verbose)
    evaluate_flink_class_dataset("WESAD", exp_path, n_jobs, verbose)
    evaluate_flink_class_dataset("MIT-BIH-VE", exp_path, n_jobs, verbose)
    evaluate_flink_class_dataset("MIT-BIH-Arr", exp_path, n_jobs, verbose)
    evaluate_flink_class_dataset("SleepDB", exp_path, n_jobs, verbose)
