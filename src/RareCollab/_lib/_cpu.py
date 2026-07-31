"""Shared CPU detection, so Setup, Features and the RNA worker agree."""

import os


def get_available_cpus():
    """
    How many cores this process may actually use.

    sched_getaffinity respects cgroups, taskset and SLURM allocations, which
    os.cpu_count() does not - on a shared node the latter reports the whole
    machine.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1