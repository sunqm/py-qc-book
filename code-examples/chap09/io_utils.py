from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

def iterate_with_prefetch(tasks, loader):
    with ThreadPoolExecutor(max_workers=1) as t_exe:
        task_to_run = None
        for task in tasks:
            if task_to_run is None:
                future = t_exe.submit(loader, task)
                task_to_run = task
                continue

            data = future.result()
            future = t_exe.submit(loader, task)
            yield task_to_run, data
            task_to_run = task

        data = future.result()
        yield task_to_run, data

@contextmanager
def background(fn):
    with ThreadPoolExecutor(max_workers=1) as t_exe:
        future = None
        def bg_worker(*args, **kwargs):
            nonlocal future
            if future is not None:
                future.result()
            future = t_exe.submit(fn, *args, **kwargs)
            return future

        try:
            yield bg_worker
        finally:
            if future is not None:
                future.result()
