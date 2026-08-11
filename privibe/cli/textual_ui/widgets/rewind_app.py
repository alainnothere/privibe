from __future__ import annotations

from enum import StrEnum, auto
import random
from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Input, Static

from privibe.cli.commands import ALT_KEY
from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput


class _RewindAction(StrEnum):
    EDIT_AND_RESTORE = auto()
    EDIT_ONLY = auto()


class RewindApp(Container):
    """Bottom panel widget for rewind mode actions.

    Restoring files overwrites and deletes on disk with no undo, so that option
    is never the default and never a single keystroke: it is reached
    deliberately and then gated behind a typed confirmation code. Number-key
    shortcuts are deliberately absent for the same reason.
    """

    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
    ]

    class RewindWithRestore(Message):
        """User chose to edit the message and restore files."""

    class RewindWithoutRestore(Message):
        """User chose to edit the message without restoring files."""

    def __init__(self, message_preview: str, *, has_file_changes: bool) -> None:
        super().__init__(id="rewind-app")
        self._message_preview = message_preview
        self._has_file_changes = has_file_changes
        self.selected_option = 0
        self.option_widgets: list[Static] = []
        self._title_widget: NoMarkupStatic | None = None
        self._confirm_widget: NoMarkupStatic | None = None
        self._confirm_input: Input | None = None
        self._help_widget: NoMarkupStatic | None = None
        self._confirm_code: str | None = None
        self._options = self._build_options()

    def _build_options(self) -> list[tuple[str, _RewindAction]]:
        """Safe option first: it is what Enter hits when the panel opens."""
        edit_only_label = (
            "Edit without restoring files"
            if self._has_file_changes
            else "Edit message from here"
        )
        options: list[tuple[str, _RewindAction]] = [
            (edit_only_label, _RewindAction.EDIT_ONLY)
        ]
        if self._has_file_changes:
            options.append((
                "Edit & restore files to this point",
                _RewindAction.EDIT_AND_RESTORE,
            ))
        return options

    @property
    def has_file_changes(self) -> bool:
        return self._has_file_changes

    @property
    def is_confirming(self) -> bool:
        return self._confirm_code is not None

    def update_preview(self, message_preview: str) -> None:
        # Browsing to another message while a confirmation is pending would let
        # the typed code apply to a target the user is no longer looking at.
        self.cancel_confirm()
        self._message_preview = message_preview
        if self._title_widget is not None:
            self._title_widget.update(f"Rewind to: {message_preview[:80]}")

    def compose(self) -> ComposeResult:
        with Vertical(id="rewind-content"):
            self._title_widget = NoMarkupStatic(
                f"Rewind to: {self._message_preview[:80]}", classes="rewind-title"
            )
            yield self._title_widget
            yield NoMarkupStatic("")
            for _ in range(len(self._options)):
                widget = NoMarkupStatic("", classes="rewind-option")
                self.option_widgets.append(widget)
                yield widget
            self._confirm_widget = NoMarkupStatic("", classes="rewind-confirm")
            self._confirm_widget.display = False
            yield self._confirm_widget
            self._confirm_input = VscodeCompatInput(
                placeholder="00", classes="rewind-confirm-input"
            )
            self._confirm_input.display = False
            yield self._confirm_input
            yield NoMarkupStatic("")
            self._help_widget = NoMarkupStatic(
                self._help_text(confirming=False), classes="rewind-help"
            )
            yield self._help_widget

    @staticmethod
    def _help_text(*, confirming: bool) -> str:
        # Built at call time, not import time: ALT_KEY is patched in tests.
        if confirming:
            return "Type the number  Enter confirm  ESC back to options"
        return (
            f"{ALT_KEY}+↑↓ or Ctrl+P/N browse messages  ↑↓ pick option  "
            "Enter confirm  ESC cancel"
        )

    async def on_mount(self) -> None:
        self._update_options()
        self.focus()

    def _update_options(self) -> None:
        for idx, ((text, _action), widget) in enumerate(
            zip(self._options, self.option_widgets, strict=True)
        ):
            is_selected = idx == self.selected_option
            cursor = "› " if is_selected else "  "
            option_text = f"{cursor}{idx + 1}. {text}"

            widget.update(option_text)

            widget.remove_class("rewind-cursor-selected")
            widget.remove_class("rewind-option-unselected")

            if is_selected:
                widget.add_class("rewind-cursor-selected")
            else:
                widget.add_class("rewind-option-unselected")

    def _option_count(self) -> int:
        return len(self._options)

    def action_move_up(self) -> None:
        if self.is_confirming:
            return
        self.selected_option = (self.selected_option - 1) % self._option_count()
        self._update_options()

    def action_move_down(self) -> None:
        if self.is_confirming:
            return
        self.selected_option = (self.selected_option + 1) % self._option_count()
        self._update_options()

    def action_select(self) -> None:
        if self.is_confirming:
            return
        self._handle_selection(self.selected_option)

    def _handle_selection(self, option: int) -> None:
        _, action = self._options[option]
        match action:
            case _RewindAction.EDIT_AND_RESTORE:
                self._enter_confirm()
            case _RewindAction.EDIT_ONLY:
                self.post_message(self.RewindWithoutRestore())

    # -- Typed confirmation for the destructive option -------------------------

    def _enter_confirm(self) -> None:
        self._new_confirm_code(retry=False)
        self._set_confirm_visible(visible=True)
        if self._confirm_input is not None:
            self._confirm_input.value = ""
            self._confirm_input.focus()

    def _new_confirm_code(self, *, retry: bool) -> None:
        self._confirm_code = str(random.randint(10, 99))
        if self._confirm_widget is not None:
            lead = (
                "Wrong number. Here is a new one."
                if retry
                else "This overwrites and deletes files on disk. It cannot be undone."
            )
            self._confirm_widget.update(
                f"{lead}\nType {self._confirm_code} and press Enter to restore files."
            )

    def _set_confirm_visible(self, *, visible: bool) -> None:
        for widget in self.option_widgets:
            widget.display = not visible
        if self._confirm_widget is not None:
            self._confirm_widget.display = visible
        if self._confirm_input is not None:
            self._confirm_input.display = visible
        if self._help_widget is not None:
            self._help_widget.update(self._help_text(confirming=visible))

    def cancel_confirm(self) -> bool:
        """Return to the option list. True if a confirmation was pending."""
        if self._confirm_code is None:
            return False
        self._confirm_code = None
        if self._confirm_input is not None:
            self._confirm_input.value = ""
        self._set_confirm_visible(visible=False)
        self._update_options()
        self.focus()
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._confirm_code is None or self._confirm_input is None:
            return
        if self._confirm_input.value.strip() == self._confirm_code:
            self._confirm_code = None
            self.post_message(self.RewindWithRestore())
            return
        # A fresh number, so a mistyped guess never gets a second attempt at the
        # same target. Getting here means the user was not reading.
        self._confirm_input.value = ""
        self._new_confirm_code(retry=True)

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self._refocus_if_needed)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        self.call_after_refresh(self._refocus_if_needed)

    def _refocus_if_needed(self) -> None:
        confirm_input = self._confirm_input
        if self.has_focus or (confirm_input is not None and confirm_input.has_focus):
            return
        if self.is_confirming and confirm_input is not None:
            confirm_input.focus()
            return
        self.focus()
