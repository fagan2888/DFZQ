
# START
import multiprocessing
import numpy as np
cc = np.array_split(codes, 30)
pool = multiprocessing.Pool(30)
res = []
for c in cc:
    res.append(pool.apply_async(job_factors_foo, (c, EP, )))
for r in res:
    r.get()
# END
