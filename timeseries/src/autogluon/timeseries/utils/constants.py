import joblib.externals.loky
from joblib import cpu_count

# By default, joblib w/ loky backend kills processes that take >300MB of RAM assuming that this is caused by a memory
# leak. This leads to problems for some memory-hungry models like AutoARIMA/Theta.
# This monkey patch removes this undesired behavior
joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)

# We use the same default n_jobs across AG-TS to ensure that Joblib reuses the process pool
AG_DEFAULT_N_JOBS = max(cpu_count(only_physical_cores=True), 1)
