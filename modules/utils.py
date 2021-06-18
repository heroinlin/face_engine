import time


def compute_time(func, args, run_num=100):
    start_time = time.time()
    for i in range(run_num):
        func(*args)
    end_time = time.time()
    avg_run_time = (end_time - start_time)*1000/run_num
    return avg_run_time
