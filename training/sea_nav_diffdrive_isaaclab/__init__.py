# SPDX-License-Identifier: MIT
"""SEA-owned DashGo differential-drive Isaac Lab package (CPU-import-safe core).

Modules here must not import Isaac, ROS, or DashGo packages at module import
time; target-stack factories fail closed with explicit blocked errors instead
of mocking.  This package never depends on the historical Go2 adapter.
"""

__version__ = "0.1.0"
