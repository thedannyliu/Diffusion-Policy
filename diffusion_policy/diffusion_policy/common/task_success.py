"""Aggregate environment success flags without counting time-limit termination."""


def episode_task_success(info):
    """Read the full info history returned by MultiStepWrapper.get_infos().

    PushT's `success` flag means coverage exceeds its task threshold. Wrapper
    `done` flags also include time limits and must not be used as success.
    Missing or empty histories indicate an invalid evaluation, not a failure.
    """
    flags = info.get("success")
    if flags is None or len(flags) == 0:
        raise ValueError("Task success history is missing; cannot compute success rate")
    return any(bool(flag) for flag in flags)
