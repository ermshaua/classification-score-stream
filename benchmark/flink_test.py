import logging
import sys

from src.clazz.flink import ClaSSProcessWindowFunction
from src.utils import load_dataset

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import CountSlidingWindowAssigner

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

    # Apply the ClaSSProcessWindowFunction
    output_stream = input_stream \
        .key_by(lambda x: 0) \
        .window(CountSlidingWindowAssigner.of(1, 1)) \
        .process(ClaSSProcessWindowFunction(n_timepoints=10_000, n_prerun=min(10_000, ts.shape[0])), output_type=Types.INT())

    # Write the output data stream
    output_stream.print()

    # Execute the job
    env.execute("Simple ClaSS Flink Test")
