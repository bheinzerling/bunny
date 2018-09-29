import multiprocessing as mp
import threading as th
from tqdm import tqdm
from tqdm._utils import _term_move_up


up = _term_move_up()


# Create global parallelism locks to avoid racing issues with parallel bars
# works only if fork available (Linux, MacOSX, but not on Windows)
try:
    mp_lock = mp.RLock()  # multiprocessing lock
except ImportError:  # pragma: no cover
    mp_lock = None
except OSError:  # pragma: no cover
    mp_lock = None
try:
    th_lock = th.RLock()  # thread lock
except OSError:  # pragma: no cover
    th_lock = None


class TqdmDefaultWriteLock(object):
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """
    def __init__(self):
        global mp_lock, th_lock
        self.locks = [lk for lk in [mp_lock, th_lock] if lk is not None]

    def acquire(self):
        for lock in self.locks:
            lock.acquire()

    def release(self):
        for lock in self.locks[::-1]:  # Release in inverse order of acquisition
            lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


class bunny(tqdm):
    monitor_interval = 10  # set to 0 to disable the thread
    monitor = None
    _lock = TqdmDefaultWriteLock()

    def __init__(self, iterable, **kwargs):
        super().__init__(iterable, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
        else:
            mininterval = self.mininterval
            maxinterval = self.maxinterval
            miniters = self.miniters
            dynamic_miniters = self.dynamic_miniters
            last_print_t = self.last_print_t
            last_print_n = self.last_print_n
            n = self.n
            smoothing = self.smoothing
            avg_time = self.avg_time
            _time = self._time

            try:
                sp = self.sp
            except AttributeError:
                raise TqdmDeprecationWarning("""\
Please use `tqdm_gui(...)` instead of `tqdm(..., gui=True)`
""", fp_write=getattr(self.fp, 'write', sys.stderr.write))

            tqdm.write("\r" + " " * self.ncols + "\n" * 8)  # make space for bunny

            for obj in iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1
                # check counter first to avoid calls to time()
                if n - last_print_n >= self.miniters:
                    miniters = self.miniters  # watch monitoring thread changes
                    delta_t = _time() - last_print_t
                    if delta_t >= mininterval:
                        cur_t = _time()
                        delta_it = n - last_print_n
                        # EMA (not just overall average)
                        if smoothing and delta_t and delta_it:
                            avg_time = delta_t / delta_it \
                                if avg_time is None \
                                else smoothing * delta_t / delta_it + \
                                (1 - smoothing) * avg_time
                            self.avg_time = avg_time

                        self.n = n
                        with self._lock:
                            if self.pos:
                                self.moveto(abs(self.pos))
                            # Print bar update
                            sp(self.__repr__())
                            if self.pos:
                                self.moveto(-abs(self.pos))

                        # If no `miniters` was specified, adjust automatically
                        # to the max iteration rate seen so far between 2 prints
                        if dynamic_miniters:
                            if maxinterval and delta_t >= maxinterval:
                                # Adjust miniters to time interval by rule of 3
                                if mininterval:
                                    # Set miniters to correspond to mininterval
                                    miniters = delta_it * mininterval / delta_t
                                else:
                                    # Set miniters to correspond to maxinterval
                                    miniters = delta_it * maxinterval / delta_t
                            elif smoothing:
                                # EMA-weight miniters to converge
                                # towards the timeframe of mininterval
                                miniters = smoothing * delta_it * \
                                    (mininterval / delta_t
                                     if mininterval and delta_t else 1) + \
                                    (1 - smoothing) * miniters
                            else:
                                # Maximum nb of iterations between 2 prints
                                miniters = max(miniters, delta_it)

                        # Store old values for next call
                        self.n = self.last_print_n = last_print_n = n
                        self.last_print_t = last_print_t = cur_t
                        self.miniters = miniters

                tqdm.write(up * 9)  # move cursor up
                if self.total:
                    # move bunny
                    offset = " " * int(n / self.total * (self.ncols - 40))
                else:
                    offset = ""
                tqdm.write(offset + '|￣￣￣￣￣￣￣￣|')
                tqdm.write(offset + '|    TRAINING    |') 
                tqdm.write(offset + '|     epoch      |')
                tqdm.write(offset + f'|   {obj:>6}       |')  
                tqdm.write(offset + '| ＿＿＿_＿＿＿＿|') 
                tqdm.write(offset + ' (\__/) ||') 
                tqdm.write(offset + ' (•ㅅ•) || ')
                tqdm.write(offset + ' / 　 づ')

            # Closing the progress bar.
            # Update some internal variables for close().
            self.last_print_n = last_print_n
            self.n = n
            self.miniters = miniters
            self.close()
