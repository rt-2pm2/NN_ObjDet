import time
from sys import float_info as flt_info 

class TimeMeas:
    """ 
    This class provides methods to perform timing measurements and extract statistics 
    """
    clock = time.CLOCK_MONOTONIC

    # Initialize the class
    def __init__(self):
        self._start = 0
        self._stop = 0
        self._elapsed = 0
        self._avg_elapsed = 0
        self._max_elapsed = 0
        self._min_elapsed = flt_info.max
        self._cnt_elapsed = int(0)

        self._avg_period = 0
        self._max_period = 0
        self._old_t = 0
        self._nTicks = int(0)

    # Start the timer
    def start(self):
        self._start = time.clock_gettime(self.clock)

    # Stop the timer and return the elapsed time
    def stop(self):
        self._stop = time.clock_gettime(self.clock)
        self._elapsed = self._stop - self._start
        elapsed = self._elapsed

        # Statistics
        K = self._cnt_elapsed
        self._avg_elapsed = self._avg_elapsed * (K/(K + 1)) + elapsed/ (K + 1)
        if (elapsed > self._max_elapsed):
            self._max_elapsed = elapsed
        if (elapsed < self._min_elapsed):
            self._min_elapsed = elapsed

        self._cnt_elapsed += 1

        return elapsed

    # Update the number of ticks and evaluate the periodicity
    def tick(self):
        # Update ticks
        K = self._nTicks

        # Measure time
        curr_t = time.clock_gettime(self.clock)  
        if (K >= 1):
            M = K - 1
            curr_period = (curr_t - self._old_t)
            # Statistics
            self._avg_period = self._avg_period * (M/ (M + 1)) + curr_period / (M + 1)
            if (curr_period > self._max_period):
                self._max_period = curr_period

        self._nTicks += 1
        self._old_t = curr_t

    # Measure the occurrence frequency
    def getfreq(self):
        if (self._avg_period > 0):
            return 1.0/self._avg_period

    def getPeriod(self):
        return self._avg_period

    # Reset the timer
    def reset(self):
        self._start = 0
        self._nTicks = 0

    
