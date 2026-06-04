"""Utilities package. Re-exports all public and test-used symbols from submodules.

Import read_safe/read_safe_async from privibe.core.utils.io and create_slug from
privibe.core.utils.slug when needed to avoid circular imports with config.
"""

from __future__ import annotations

from privibe.core.utils.concurrency import (
    AsyncExecutor,
    ConversationLimitException,
    run_sync,
)
from privibe.core.utils.display import compact_reduction_display
from privibe.core.utils.http import get_server_url_from_api_base, get_user_agent
from privibe.core.utils.matching import name_matches
from privibe.core.utils.paths import is_dangerous_directory
from privibe.core.utils.platform import is_windows
from privibe.core.utils.retry import async_generator_retry, async_retry
from privibe.core.utils.tags import (
    CANCELLATION_TAG,
    CONTEXT_REFRESH_TAG,
    KNOWN_TAGS,
    TOOL_ERROR_TAG,
    VIBE_STOP_EVENT_TAG,
    VIBE_WARNING_TAG,
    CancellationReason,
    TaggedText,
    get_user_cancellation_message,
    is_user_cancellation_event,
)
from privibe.core.utils.time import utc_now

__all__ = [
    "CANCELLATION_TAG",
    "CONTEXT_REFRESH_TAG",
    "KNOWN_TAGS",
    "TOOL_ERROR_TAG",
    "VIBE_STOP_EVENT_TAG",
    "VIBE_WARNING_TAG",
    "AsyncExecutor",
    "CancellationReason",
    "ConversationLimitException",
    "TaggedText",
    "async_generator_retry",
    "async_retry",
    "compact_reduction_display",
    "get_server_url_from_api_base",
    "get_user_agent",
    "get_user_cancellation_message",
    "is_dangerous_directory",
    "is_user_cancellation_event",
    "is_windows",
    "name_matches",
    "run_sync",
    "utc_now",
]
