import os
import re


def get_continuation_fname(CONT, logdir):
    if CONT.lower() == "last":
        # Take last
        files = [f for f in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, f))]
        epochNames = [re.match(f"epoch(\d+)_step\d+.*\.pt$", f) for f in files]
        epochs = [match.group(1) for match in filter(lambda x: x is not None, epochNames)]
        maxEpoch = sorted(epochs, reverse=True, key=int)[0]

        fPrefix = f'epoch{maxEpoch}_step'
        fnames = [re.match(f"({re.escape(fPrefix)}\d+).*\.pt$", f) for f in files]
        # Take first if multiple matches (unlikely)
        fname = ([f for f in fnames if f is not None][0]).group(1)

        epoch = int(maxEpoch)

    else:

        # parametrized by the epoch
        fPrefix = f'epoch{CONT}_step'
        files = [f for f in os.listdir(logdir) if os.path.isfile(os.path.join(logdir, f))]
        fnames = [re.match(f"({re.escape(fPrefix)}\d+).*\.pt$", f) for f in files]
        # Take first if multiple matches (unlikely)
        fname = ([f for f in fnames if f is not None][0]).group(1)

        epoch = int(CONT)

    return fname, epoch
