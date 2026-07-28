# A single, shared Limiter instance.
#
# Why this lives in its own module instead of main.py: main.py imports the
# router from app/api/routes.py, and routes.py needs to import `limiter` to
# decorate /chat with @limiter.limit(...). If `limiter` were defined in
# main.py, that would create a circular import (main -> routes -> main).
# Putting it in a standalone module with no dependency on either file breaks
# the cycle — both main.py and routes.py import from here instead.
from slowapi import Limiter
from slowapi.util import get_remote_address

# key_func decides what counts as "one client" for the purposes of counting
# requests. get_remote_address reads the caller's IP from the request. That
# means each IP gets its own independent 60/min (or whatever settings.rate_limit
# is) budget, tracked in-memory by slowapi.
limiter = Limiter(key_func=get_remote_address)
