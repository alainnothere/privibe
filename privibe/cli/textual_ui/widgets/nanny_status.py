from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from privibe.core.nanny import NannyListener, NannyManager, NannyState

FRAMES = "◐◓◑◒"
ANIMATION_INTERVAL = 0.15


class NannyStatus(NannyListener, Static):
    """Loading-area line shown while the nanny is reading a call or
    answering the human; empty otherwise. Attached to whichever nanny the
    app currently runs, since the nanny can be turned on and off mid-session.
    """

    state = reactive(NannyState.IDLE)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._nanny: NannyManager | None = None
        self._timer: Timer | None = None
        self._frame = 0

    def attach(self, nanny: NannyManager | None) -> None:
        if self._nanny is not None:
            self._nanny.remove_listener(self)
        self._nanny = nanny
        if nanny is not None:
            nanny.add_listener(self)
            self.state = nanny.state
        else:
            self.state = NannyState.IDLE

    def on_unmount(self) -> None:
        self.attach(None)

    def on_nanny_state_change(self, state: NannyState) -> None:
        # Listeners fire from the agent task; reactives are safe to set from
        # there as long as the app loop is the same loop, which it is.
        self.state = state

    def watch_state(self, new_state: NannyState) -> None:
        self._stop_timer()
        match new_state:
            case NannyState.IDLE:
                self.update("")
            case NannyState.REVIEWING | NannyState.TALKING:
                self._frame = 0
                self._tick()
                self._timer = self.set_interval(ANIMATION_INTERVAL, self._tick)

    def _tick(self) -> None:
        char = FRAMES[self._frame % len(FRAMES)]
        label = (
            "nanny reviewing"
            if self.state is NannyState.REVIEWING
            else "nanny answering"
        )
        self.update(f"[bold magenta]{char}[/bold magenta] {label} ")
        self._frame += 1

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
