import numpy as np

class Window:
    def __init__(self, window_list=[]):
        """
        Parameters
        ----------
        window_list : list of tuples, optional
            Each tuple should be in the form (start_time, end_time) where the 
            times are Astropy Time objects. Default is an empty list.
    
        Attributes
        ----------
        list : list of tuples
        
        num : int
            Number of windows in the list
        """
        self.list = window_list
        self.num = len(window_list)

    
    def update_num(self):
        """ 
        Updates the num attribute to reflect the current length of list. 
        """
        self.num = len(self.list)

    
    def __add__(self, other):
        return Window(self.list + other.list)

    
    def includes(self, time):
        """
        Checks if any of the windows includes the given `time` in its range.
        
        Parameters
        ----------
        time : astropy.time.Time
            The time to check against the window ranges.
        
        Returns
        -------
        bool
            `True` if the time is within any window, `False` otherwise.
        """
        for window in self.list:
            start_time, end_time = window
            if start_time <= time <= end_time:
                return True
        return False

    
    def get_start_time(self):
        """
        Returns the `start_time` of the earliest full window.
        """
        if self.num == 1:
            start_time = self.list[0][0]
        elif self.num > 1:
            start_time = self.list[1][0]
        else:
            raise ValueError("The are no windows.")
            
        return start_time

    
    def get_timespan(self):
        """
        Returns the total timespan (in seconds) covered by all the windows.
        """
        timespan = sum((end_time - start_time).sec for start_time, end_time in self.list)
        return timespan

    
    def get_each_timespan(self):
        """
        Returns a list of timespans (in seconds) for each individual window.
        """
        timespans = [(end_time - start_time).sec for start_time, end_time in self.list]
        return np.array(timespans)

    
    def sort(self):
        """
        Sorts the windows in chronological order (by each of their start_time). 
        """
        self.list.sort(key=lambda window: window[0])
        return self

    
    def filter(self, min_timespan):
        """
        Filters out windows with timespan less than `min_timespan` (in seconds). 
        """
        self.list = [window for window in self.list if (window[1] - window[0]).sec >= min_timespan]
        self.update_num()
        return self

    
    def merge(self):
        """
        Sorts and merges overlapping windows.
        """
        if self.num != 0:
            # Sort windows by start time
            self.sort()
            
            # Initialize merged list with the first window
            merged = [self.list[0]]
            
            for current in self.list[1:]:
                last = merged[-1]
                # Check if current window overlaps with the last merged window
                if current[0] <= last[1]: 
                    # If there's an overlap
                    # merge the windows by updating the end time of the last merged window
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    # If no overlap, add the current window to the merged list
                    merged.append(current)
                    
            self.list = merged
            self.update_num()
            
        return self