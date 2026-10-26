from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.tool_widgets import (
    get_approval_widget,
    half_viewport_cap,
)
from privibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput
from privibe.core.config import VibeConfig
from privibe.core.tools.permissions import RequiredPermission

if TYPE_CHECKING:
    from privibe.core.nanny import NannyReview

# Option positions. The first four are the historical prompt and keep their
# keys and messages; the last two exist only when a nanny review is attached.
OPTION_YES = 0
OPTION_ALWAYS = 1
OPTION_NO = 2
OPTION_DENY = 3
OPTION_REFUSE = 4
OPTION_TALK = 5

HELP_PLAIN = "↑↓ navigate  Enter select  ESC reject"
HELP_NANNY = "↑↓ navigate  Enter select  ESC reject  (r refuse  t talk to the nanny)"
HELP_TALK = "Enter send  ESC back to the options"


class ApprovalApp(Container):
    can_focus = True
    can_focus_children = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "select_1", "Yes", show=False),
        Binding("y", "select_1", "Yes", show=False),
        Binding("2", "select_2", "Always Tool Session", show=False),
        Binding("3", "select_3", "No", show=False),
        Binding("n", "select_3", "No", show=False),
        Binding("4", "select_4", "Never Session", show=False),
        Binding("d", "select_4", "Never Session", show=False),
        Binding("5", "select_5", "Refuse", show=False),
        Binding("r", "select_5", "Refuse", show=False),
        Binding("6", "select_6", "Talk to the nanny", show=False),
        Binding("t", "select_6", "Talk to the nanny", show=False),
    ]

    class ApprovalGranted(Message):
        def __init__(self, tool_name: str, tool_args: BaseModel) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args

    class ApprovalGrantedAlwaysTool(Message):
        def __init__(
            self,
            tool_name: str,
            tool_args: BaseModel,
            required_permissions: list[RequiredPermission],
        ) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args
            self.required_permissions = required_permissions

    class ApprovalRejected(Message):
        def __init__(self, tool_name: str, tool_args: BaseModel) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args

    class ApprovalRejectedAlwaysTool(Message):
        def __init__(
            self,
            tool_name: str,
            tool_args: BaseModel,
            required_permissions: list[RequiredPermission],
        ) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args
            self.required_permissions = required_permissions

    class NannyRefused(Message):
        """Refuse the call with the nanny's reason as the worker's feedback."""

        def __init__(self, tool_name: str, tool_args: BaseModel) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args

    class NannyTalk(Message):
        """One line from the human to the nanny; the app answers through
        show_nanny_reply.
        """

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(
        self,
        tool_name: str,
        tool_args: BaseModel,
        config: VibeConfig,
        required_permissions: list[RequiredPermission] | None = None,
        review: NannyReview | None = None,
    ) -> None:
        super().__init__(id="approval-app")
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.config = config
        self.required_permissions = required_permissions or []
        self.review = review
        self.selected_option = 0
        self.talk_mode = False
        self.talk_pending = False
        self.content_container: Vertical | None = None
        self.title_widget: Static | None = None
        self.tool_info_container: Vertical | None = None
        self.option_widgets: list[Static] = []
        self.help_widget: Static | None = None
        self.nanny_widget: Static | None = None
        self.nanny_log_widget: Static | None = None
        self.talk_row: Horizontal | None = None
        self.talk_input: Input | None = None
        self._nanny_log: list[str] = []

    @property
    def has_review(self) -> bool:
        return self.review is not None

    @property
    def option_count(self) -> int:
        return 6 if self.has_review else 4

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-options"):
            yield NoMarkupStatic("")
            for _ in range(self.option_count):
                widget = NoMarkupStatic("", classes="approval-option")
                self.option_widgets.append(widget)
                yield widget
            self.talk_row = Horizontal(classes="approval-talk-row")
            with self.talk_row:
                yield NoMarkupStatic("To the nanny: ", classes="approval-talk-prefix")
                self.talk_input = VscodeCompatInput(
                    placeholder="Explain, confirm, or set an exception...",
                    classes="approval-talk-input",
                )
                yield self.talk_input
            yield NoMarkupStatic("")
            self.help_widget = NoMarkupStatic(
                HELP_NANNY if self.has_review else HELP_PLAIN, classes="approval-help"
            )
            yield self.help_widget

        with Vertical(id="approval-content"):
            self.title_widget = NoMarkupStatic(
                f"⚠ {self.tool_name} command", classes="approval-title"
            )
            yield self.title_widget

            with VerticalScroll(classes="approval-tool-info-scroll"):
                self.tool_info_container = Vertical(
                    classes="approval-tool-info-container"
                )
                yield self.tool_info_container

            if self.has_review:
                self.nanny_widget = NoMarkupStatic(
                    self._nanny_headline(), classes="approval-nanny"
                )
                yield self.nanny_widget
                self.nanny_log_widget = NoMarkupStatic("", classes="approval-nanny-log")
                yield self.nanny_log_widget

    def _nanny_headline(self) -> str:
        review = self.review
        if review is None:
            return ""
        if review.failed:
            return f"nanny: {review.notice}"
        text = f"nanny refused: {review.reason}"
        if review.notice:
            text += f"\n({review.notice})"
        return text

    async def on_mount(self) -> None:
        # Cells-based cap (was max-height: 50vh in tcss; see
        # half_viewport_cap). Re-applied from VibeApp.on_resize.
        self.query_one(
            ".approval-tool-info-scroll"
        ).styles.max_height = half_viewport_cap(self.app.size.height)
        if self.talk_row is not None:
            self.talk_row.display = False
        await self._update_tool_info()
        self._update_options()
        self.focus()

    async def _update_tool_info(self) -> None:
        if not self.tool_info_container:
            return

        approval_widget = get_approval_widget(self.tool_name, self.tool_args)
        await self.tool_info_container.remove_children()
        await self.tool_info_container.mount(approval_widget)

    def _option_texts(self) -> list[tuple[str, str]]:
        if self.required_permissions:
            labels = ", ".join(rp.label for rp in self.required_permissions)
            always_text = f"Yes and always allow for this session: {labels}"
            never_text = f"No and deny for this session: {labels}"
        else:
            always_text = f"Yes and always allow {self.tool_name} for this session"
            never_text = f"No and deny {self.tool_name} for this session"

        options = [
            ("Yes", "yes"),
            (always_text, "yes"),
            ("No and tell the agent what to do instead", "no"),
            (never_text, "no"),
        ]
        if self.has_review:
            options.append(("Refuse, and tell the agent the nanny's reason", "no"))
            options.append(("Talk to the nanny", "talk"))
        return options

    def _update_options(self) -> None:
        for idx, ((text, color_type), widget) in enumerate(
            zip(self._option_texts(), self.option_widgets, strict=True)
        ):
            is_selected = idx == self.selected_option and not self.talk_mode

            cursor = "› " if is_selected else "  "
            option_text = f"{cursor}{idx + 1}. {text}"

            widget.update(option_text)

            widget.remove_class("approval-cursor-selected")
            widget.remove_class("approval-option-selected")
            widget.remove_class("approval-option-yes")
            widget.remove_class("approval-option-no")
            widget.remove_class("approval-option-talk")

            color_class = f"approval-option-{color_type}"
            if is_selected:
                widget.add_class("approval-cursor-selected")
            else:
                widget.add_class("approval-option-selected")
            widget.add_class(color_class)

    # ------------------------------------------------------------------
    # Talk mode
    # ------------------------------------------------------------------

    def _enter_talk_mode(self) -> None:
        if not self.has_review or self.talk_row is None or self.talk_input is None:
            return
        self.talk_mode = True
        self.talk_row.display = True
        if self.help_widget:
            self.help_widget.update(HELP_TALK)
        self._update_options()
        self.talk_input.focus()

    def _leave_talk_mode(self) -> None:
        self.talk_mode = False
        if self.talk_row is not None:
            self.talk_row.display = False
        if self.help_widget:
            self.help_widget.update(HELP_NANNY if self.has_review else HELP_PLAIN)
        self._update_options()
        self.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if not self.talk_mode or self.talk_pending or self.talk_input is None:
            return
        text = self.talk_input.value.strip()
        if not text:
            return
        self.talk_input.value = ""
        self.talk_pending = True
        self.talk_input.disabled = True
        self._append_nanny_log(f"you: {text}")
        self._append_nanny_log("nanny is thinking...")
        self.post_message(self.NannyTalk(text))

    def show_nanny_reply(self, reply: str) -> None:
        """Called by the app once the nanny answered a NannyTalk."""
        if self._nanny_log and self._nanny_log[-1] == "nanny is thinking...":
            self._nanny_log.pop()
        self._append_nanny_log(f"nanny: {reply}")
        self.talk_pending = False
        if self.talk_input is not None:
            self.talk_input.disabled = False
            if self.talk_mode:
                self.talk_input.focus()

    def _append_nanny_log(self, line: str) -> None:
        self._nanny_log.append(line)
        if self.nanny_log_widget is not None:
            self.nanny_log_widget.update("\n".join(self._nanny_log))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_move_up(self) -> None:
        if self.talk_mode:
            return
        self.selected_option = (self.selected_option - 1) % self.option_count
        self._update_options()

    def action_move_down(self) -> None:
        if self.talk_mode:
            return
        self.selected_option = (self.selected_option + 1) % self.option_count
        self._update_options()

    def action_select(self) -> None:
        if self.talk_mode:
            return
        self._handle_selection(self.selected_option)

    def _select(self, option: int) -> None:
        if self.talk_mode or option >= self.option_count:
            return
        self.selected_option = option
        self._handle_selection(option)

    def action_select_1(self) -> None:
        self._select(OPTION_YES)

    def action_select_2(self) -> None:
        self._select(OPTION_ALWAYS)

    def action_select_3(self) -> None:
        self._select(OPTION_NO)

    def action_select_4(self) -> None:
        self._select(OPTION_DENY)

    def action_select_5(self) -> None:
        self._select(OPTION_REFUSE)

    def action_select_6(self) -> None:
        self._select(OPTION_TALK)

    def action_reject(self) -> None:
        """Escape: in talk mode it returns to the options; otherwise it is
        the historical reject.
        """
        if self.talk_mode:
            if not self.talk_pending:
                self._leave_talk_mode()
            return
        self.selected_option = OPTION_NO
        self._handle_selection(OPTION_NO)

    def _handle_selection(self, option: int) -> None:
        match option:
            case 0:
                self.post_message(
                    self.ApprovalGranted(
                        tool_name=self.tool_name, tool_args=self.tool_args
                    )
                )
            case 1:
                self.post_message(
                    self.ApprovalGrantedAlwaysTool(
                        tool_name=self.tool_name,
                        tool_args=self.tool_args,
                        required_permissions=self.required_permissions,
                    )
                )
            case 2:
                self.post_message(
                    self.ApprovalRejected(
                        tool_name=self.tool_name, tool_args=self.tool_args
                    )
                )
            case 3:
                self.post_message(
                    self.ApprovalRejectedAlwaysTool(
                        tool_name=self.tool_name,
                        tool_args=self.tool_args,
                        required_permissions=self.required_permissions,
                    )
                )
            case 4:
                self.post_message(
                    self.NannyRefused(
                        tool_name=self.tool_name, tool_args=self.tool_args
                    )
                )
            case 5:
                self._enter_talk_mode()

    def on_blur(self, event: events.Blur) -> None:
        if self.talk_mode and self.talk_input is not None and self.talk_input.has_focus:
            return
        self.call_after_refresh(self._regain_focus)

    def _regain_focus(self) -> None:
        if self.talk_mode and self.talk_input is not None:
            if not self.talk_input.has_focus:
                self.talk_input.focus()
            return
        self.focus()
