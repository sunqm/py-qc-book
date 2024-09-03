from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

def _ensure(task):
    if isinstance(task, Future):
        return task.result()
    else:
        return task

def df_scheduler(queue, fn, tasks, args):
    '''Depth-first scheduler'''
    def fn_with_future(task):
        return fn(_ensure(task), *args)
    return (queue.submit(fn_with_future, task) for task in tasks)

def bf_scheduler(queue, fn, tasks, args):
    '''Breadth-first scheduler'''
    tasks = deque(df_scheduler(queue, fn, tasks, args))
    while tasks:
        yield tasks.popleft()

def df_warmup_scheduler(queue, fn, tasks, args):
    '''Launches several tasks at beginning then fills task queue like df_scheduler'''
    def fn_with_future(task):
        return fn(_ensure(task), *args)

    max_workers = queue._max_workers * 2
    future_buf = deque()
    for task in tasks:
        future_buf.append(queue.submit(fn_with_future, task))
        if len(future_buf) > max_workers:
            yield future_buf.popleft()
    while future_buf:
        yield future_buf.popleft()

class LaunchInFuture(Future):
    def __init__(self, queue, fn):
        self.queue = queue
        self.fn = fn
        super().__init__()

    def submit(self, task):
        future = self.queue.submit(self.fn, task)
        future.add_done_callback(self.pass_result)

    def pass_result(self, value):
        self.set_result(value.result())

def dynamic_scheduler(queue, fn, tasks, args):
    '''Dynamic scheduler submits tasks only if they are ready to proceed'''
    def fn_with_future(task):
        return fn(_ensure(task), *args)

    max_workers = queue._max_workers * 2
    future_buf = deque()
    for task in tasks:
        # Block the submission to prevent the task queue being filled with
        # too many operations of the same type
        if len(future_buf) >= max_workers:
            future_buf.popleft().result()
        if not isinstance(task, Future):
            future = queue.submit(fn_with_future, task)
        else:
            future = LaunchInFuture(queue, fn_with_future)
            task.add_done_callback(future.submit)
        future_buf.append(future)
        yield future

def pipeline(ops, tasks, scheduler=dynamic_scheduler):
    '''
    Streams operations and tasks
    '''
    queue, fn, args = ops[0]
    if queue is None:
        with ThreadPoolExecutor() as queue:
            return pipeline([(queue, fn, args), *ops[1:]], tasks, scheduler)

    if len(ops) == 1:
        # Call list() to traverse through the entire schduler generator which
        # will launch all tasks and their upstream tasks.
        return [_ensure(f) for f in list(scheduler(queue, fn, tasks, args))]

    return pipeline(ops[1:], scheduler(queue, fn, tasks, args), scheduler)
