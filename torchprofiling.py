# poor mans NVTX range
import torch
import time

TIME_START = dict()
TIME_SUM = dict()
AVG_COUNT = dict()

def time_start(tag, iters=10):
    global TIME_START
    if tag not in TIME_START or AVG_COUNT[tag] % iters == 0:
        torch.cuda.synchronize()
        TIME_START[tag] = time.time()
        AVG_COUNT[tag] = 0.0
        TIME_SUM[tag] = 0.0

def time_stop(tag, iters=10):
    global TIME_SUM
    global AVG_COUNT
    assert tag in TIME_START
    assert tag in AVG_COUNT
    assert tag in TIME_SUM
    if AVG_COUNT[tag] % iters == 0:
        torch.cuda.synchronize()
        TIME_SUM[tag] += (time.time() - TIME_START[tag])
        print(tag, "CURR AVG:", TIME_SUM[tag]/((AVG_COUNT[tag] // iters) + 1))
    AVG_COUNT[tag] += 1
