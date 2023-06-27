import logging
import resource
import sys
import time

sys.path.insert(0, "../")

from pyflink.common.watermark_strategy import TimestampAssigner

from src.clazz.flink import ClaSSProcessWindowFunction
from src.utils import load_dataset

import numpy as np

np.random.seed(1379)

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic, ProcessWindowFunction


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 0

    df = load_dataset("TSSB", [selection])  #
    name, w, cps, ts = df.iloc[0, :].tolist()

    # Set up the environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_stream_time_characteristic(TimeCharacteristic.ProcessingTime)
    env.set_parallelism(1)

    # Read the input data stream
    input_stream = env.from_collection(ts.tolist(), type_info=Types.FLOAT())

    window_function = ClaSSProcessWindowFunction(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]))

    # Apply the ClaSSProcessWindowFunction
    output_stream = input_stream \
        .key_by(lambda x: 0) \
        .count_window(10) \
        .process(window_function, output_type=Types.INT())

    # Write the output data stream
    output_stream.print()

    # Before execution
    start_time = time.time()
    memory_usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # In MB

    # Execute the job
    env.execute("Simple ClaSS Flink Test")

    # After execution
    end_time = time.time()
    memory_usage_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # In MB

    runtime = end_time - start_time
    memory = memory_usage_after - memory_usage_before

    print(f"Throughput: {np.round(len(ts) / runtime, 2)} data/second")
    print(f"Memory used: {np.round(memory_usage_after - memory_usage_before, 2)} MB")
