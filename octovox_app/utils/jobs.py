"""In-memory job registry for background pipeline runs.

A single process-wide dict guarded by a lock, shared by the legacy
launch-and-poll endpoints (``/api/process_poll`` + ``/api/job/<id>``).
The streaming ``/api/process`` endpoint manages its own per-request state.
"""
import threading

JOBS = {}
JOBS_LOCK = threading.Lock()
