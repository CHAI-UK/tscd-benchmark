import sys
import os
import pandas as pd
import tempfile
from pathlib import Path
from cdt.utils.R import launch_R_script

_pkg_dir = Path(__file__).resolve().parent
_scripts_dir = _pkg_dir.parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.append(str(_scripts_dir))

from algorithms.diffan.utils import np_to_csv

def cam_pruning(A, X, cutoff):
    with tempfile.TemporaryDirectory() as save_path:
        pruning_path = _pkg_dir / "pruning_R_files/cam_pruning.R"
        print(pruning_path)

        data_np = X
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(A, save_path)

        arguments = dict()
        # the R script sources its helpers relative to this directory
        arguments['{SOURCE_DIR}'] = (_pkg_dir / "pruning_R_files").as_posix()
        arguments['{PATH_DATA}'] = data_csv_path
        arguments['{PATH_DAG}'] = dag_csv_path
        arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
        arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
        arguments['{CUTOFF}'] = str(cutoff)
        arguments['{VERBOSE}'] = "FALSE" # TRUE, FALSE

        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A

        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)
    return dag
