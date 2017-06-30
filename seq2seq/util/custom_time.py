import time


def pretty_interval(start_time):
    """
    Given a previous point in time (e.g. the start of process, measured by calling time.time()), calculates
    the time elapsed since that point, divides the elapsed time days/hours/minutes/seconds, and returns that summary

    Args:
        start_time (float): the start of the interval of time, obtained by calling time.time()
    Returns:
        str: elapsed time in '##d ##h ##m ##s' format
    """
    return pretty_time(time.time() - start_time)


def pretty_time(timespan_in_seconds):
    """
    Given a length of time (measured in seconds), divides that timespan into
    days/hours/minutes/seconds and returns that summary.

    Args:
     timespan_in_seconds (float): the number of seconds in a span of time

    Returns:
        str: timespan in '##d ##h ##m ##s' format


    Examples::

        >>> print seq2seq.util.custom_time.pretty_time(426753)
        >>> 4d 22h 32m 33
    """
    seconds = abs(int(timespan_in_seconds))
    msg = []
    days, seconds = divmod(seconds, 86400)
    if days > 0:
        msg.append("%dd" % days)
    hours, seconds = divmod(seconds, 3600)
    if hours > 0:
        msg.append("%dh" % hours)
    minutes, seconds = divmod(seconds, 60)
    if minutes > 0:
        msg.append("%dm" % minutes)
    msg.append("%ds" % seconds)
    return " ".join(msg)
