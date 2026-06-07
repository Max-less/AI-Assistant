"""App-level configuration read from the environment."""

import os

# Max number of /api/chat requests a guest account may make before being asked
# to register. Counted as the guest's persisted user-role messages.
GUEST_QUERY_LIMIT = int(os.getenv("GUEST_QUERY_LIMIT", "5"))
