"""Configuration for run_xy_sweep.py.

Edit X_VALUES, Y_VALUES, and RUNS_PER_PAIR here.
The defaults use 10 values per axis: -10%, -8%, ..., +8%.
"""

BASE_X = 0.78
BASE_Y = 0.936

PERCENT_DELTAS = [-6, -4, -2, 0, 2, 4, 6]

X_VALUES = [round(BASE_X * (1.0 + pct / 100.0), 6) for pct in PERCENT_DELTAS]
Y_VALUES = [round(BASE_Y * (1.0 + pct / 100.0), 6) for pct in PERCENT_DELTAS]

RUNS_PER_PAIR = 1               # unnecessary to change right now
TEST_SIM_NUM_TRIALS = 200        # defines number of randomized trials
RUNNER_VERBOSE = True
MAX_WORKERS = None              # None -> use up to os.cpu_count() workers
TRIAL_WORKERS_PER_PAIR = None   # None -> auto; >1 splits a pair's trials across workers

OUTPUT_XML = "modified_model.xml"
RESULTS_CSV = "sweep_results.csv"
OVERWRITE_RESULTS_CSV = True
