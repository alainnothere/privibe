from __future__ import annotations

from typing import Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic


def _build_option_text(label: str, is_current: bool) -> Text:
    text = Text(no_wrap=True)
    marker = "› " if is_current else "  "
    style = "bold" if is_current else ""
    text.append(marker, style="green" if is_current else "")
    text.append(label, style=style)
    return text


class OptionPickerApp(Container):
    """Generic bottom app for picking one value for a setting.

    The `setting` tag travels back in the OptionPicked message so the app
    knows which config value the choice belongs to.
    """

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    class OptionPicked(Message):
        def __init__(self, setting: str, value: str) -> None:
            self.setting = setting
            self.value = value
            super().__init__()

    class Cancelled(Message):
        pass

    def __init__(
        self,
        setting: str,
        title: str,
        options: list[tuple[str, str]],
        current: str,
        error: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="optionpicker-app", **kwargs)
        self._setting = setting
        self._title = title
        self._options = options
        self._current = current
        self._error = error

    def compose(self) -> ComposeResult:
        options = [
            Option(_build_option_text(label, value == self._current), id=value)
            for value, label in self._options
        ]
        with Vertical(id="optionpicker-content"):
            if self._error:
                yield NoMarkupStatic(self._error, classes="optionpicker-error")
            yield NoMarkupStatic(self._title, classes="optionpicker-title")
            yield OptionList(*options, id="optionpicker-options")
            yield NoMarkupStatic(
                "↑↓ Navigate  Enter Select  Esc Cancel", classes="optionpicker-help"
            )

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        for i, (value, _) in enumerate(self._options):
            if value == self._current:
                option_list.highlighted = i
                break
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.post_message(self.OptionPicked(self._setting, event.option.id))

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())
