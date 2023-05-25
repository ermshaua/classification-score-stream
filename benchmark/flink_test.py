import logging
import os
import sys
import time
import psutil
from claspy.data_loader import load_tssb_dataset
from pyflink.datastream.connectors.file_system import FileSink

from src.clazz.flink import ClaSSProcessWindowFunction
from src.utils import load_dataset

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, SinkFunction
from pyflink.datastream.window import CountSlidingWindowAssigner


class ListSink(SinkFunction):

    def __init__(self):
        super().__init__("CollectSink")
        self.result_array = []

    def invoke(self, value):
        self.result_array.append(value)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    selection = 0

    df = load_dataset("TSSB", [selection])  #
    name, w, cps, ts = df.iloc[0, :].tolist()

    # Set up the environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    # Read the input data stream
    input_stream = env.from_collection(ts.tolist(), type_info=Types.FLOAT())

    window_function = ClaSSProcessWindowFunction(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0]))

    # Apply the ClaSSProcessWindowFunction
    output_stream = input_stream \
        .key_by(lambda x: 0) \
        .window(CountSlidingWindowAssigner.of(1, 1)) \
        .process(window_function, output_type=Types.INT())

    # Write the output data stream
    output_stream.print()

    # Before execution
    start_time = time.time()
    memory_usage_before = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # In MB

    # Execute the job
    env.execute("Simple ClaSS Flink Test")

    # After execution
    end_time = time.time()
    memory_usage_after = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # In MB

    print(f"Execution time: {end_time - start_time} seconds")
    print(f"Memory used: {memory_usage_after - memory_usage_before} MB")