import math

def getTimeLeft(sample_time, samples_left):
    time_left = (sample_time * float(samples_left))/3600
    time_left_h = math.floor(time_left)
    time_left_m = math.floor((time_left - time_left_h)*60)
    return time_left_h, time_left_m