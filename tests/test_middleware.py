from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from privibe.core.agents.models import BUILTIN_AGENTS, CHAT, AgentProfile, BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.middleware import (
    SUSPEND_TOOLS_METADATA_KEY,
    chat_agent_exit,
    chat_agent_reminder,
    plan_agent_exit,
    ConversationContext,
    MiddlewareAction,
    MiddlewarePipeline,
    ReadOnlyAgentMiddleware,
    RepeatedToolCallMiddleware,
    ResetReason,
    StepBudgetMiddleware,
    make_plan_agent_reminder,
)
from privibe.core.conversation import ConversationList
from privibe.core.types import AgentStats, FunctionCall, LLMMessage, Role, ToolCall

REMINDER = "test reminder"
EXIT_MSG = "test exit"
TARGET_AGENT = BuiltinAgentName.PLAN


def _build_middleware(
    profile_getter,
    agent_name: str = TARGET_AGENT,
    reminder: str = REMINDER,
    exit_message: str = EXIT_MSG,
) -> ReadOnlyAgentMiddleware:
    return ReadOnlyAgentMiddleware(profile_getter, agent_name, reminder, exit_message)


@pytest.fixture
def ctx(vibe_config: VibeConfig) -> ConversationContext:
    return ConversationContext(
        messages=ConversationList(), stats=AgentStats(), config=vibe_config
    )


class TestReadOnlyAgentMiddleware:
    @pytest.mark.asyncio
    async def test_injects_reminder_when_target_agent_active(
        self, ctx: ConversationContext
    ) -> None:
        middleware = _build_middleware(lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN])

        result = await middleware.before_turn(ctx)

        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == REMINDER

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_name",
        [
            BuiltinAgentName.DEFAULT,
            BuiltinAgentName.AUTO_APPROVE,
            BuiltinAgentName.ACCEPT_EDITS,
        ],
    )
    async def test_does_not_inject_when_non_target_agent(
        self, ctx: ConversationContext, agent_name: str
    ) -> None:
        middleware = _build_middleware(lambda: BUILTIN_AGENTS[agent_name])

        result = await middleware.before_turn(ctx)

        assert result.action == MiddlewareAction.CONTINUE
        assert result.message is None

    @pytest.mark.asyncio
    async def test_injects_reminder_only_once(self, ctx: ConversationContext) -> None:
        middleware = _build_middleware(lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN])

        result1 = await middleware.before_turn(ctx)
        assert result1.action == MiddlewareAction.INJECT_MESSAGE
        assert result1.message == REMINDER

        result2 = await middleware.before_turn(ctx)
        assert result2.action == MiddlewareAction.CONTINUE
        assert result2.message is None

    @pytest.mark.asyncio
    async def test_injects_exit_message_when_leaving(
        self, ctx: ConversationContext
    ) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        await middleware.before_turn(ctx)

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == EXIT_MSG

    @pytest.mark.asyncio
    async def test_reinjects_reminder_on_reentry(
        self, ctx: ConversationContext
    ) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        result1 = await middleware.before_turn(ctx)
        assert result1.action == MiddlewareAction.INJECT_MESSAGE
        assert result1.message == REMINDER

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        result2 = await middleware.before_turn(ctx)
        assert result2.action == MiddlewareAction.INJECT_MESSAGE
        assert result2.message == EXIT_MSG

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        result3 = await middleware.before_turn(ctx)
        assert result3.action == MiddlewareAction.INJECT_MESSAGE
        assert result3.message == REMINDER

    @pytest.mark.asyncio
    async def test_custom_reminder(self, ctx: ConversationContext) -> None:
        custom_reminder = "Custom reminder"
        middleware = _build_middleware(
            lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN], reminder=custom_reminder
        )

        result = await middleware.before_turn(ctx)

        assert result.message == custom_reminder

    @pytest.mark.asyncio
    async def test_custom_exit_message(self, ctx: ConversationContext) -> None:
        custom_exit = "Custom exit message"
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(
            lambda: current_profile, exit_message=custom_exit
        )

        await middleware.before_turn(ctx)

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        result = await middleware.before_turn(ctx)
        assert result.message == custom_exit

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, ctx: ConversationContext) -> None:
        middleware = _build_middleware(lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN])
        await middleware.before_turn(ctx)

        middleware.reset()

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE

    @pytest.mark.asyncio
    async def test_exit_message_fires_only_once(self, ctx: ConversationContext) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        await middleware.before_turn(ctx)

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == EXIT_MSG

        result2 = await middleware.before_turn(ctx)
        assert result2.action == MiddlewareAction.CONTINUE
        assert result2.message is None

    @pytest.mark.asyncio
    async def test_multiple_turns_after_entry(self, ctx: ConversationContext) -> None:
        middleware = _build_middleware(lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN])

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE

        for _ in range(5):
            result = await middleware.before_turn(ctx)
            assert result.action == MiddlewareAction.CONTINUE
            assert result.message is None

    @pytest.mark.asyncio
    async def test_multiple_turns_after_exit(self, ctx: ConversationContext) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        await middleware.before_turn(ctx)

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        await middleware.before_turn(ctx)

        for _ in range(5):
            result = await middleware.before_turn(ctx)
            assert result.action == MiddlewareAction.CONTINUE
            assert result.message is None

    @pytest.mark.asyncio
    async def test_rapid_toggling_multiple_cycles(
        self, ctx: ConversationContext
    ) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        for _ in range(3):
            current_profile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
            result = await middleware.before_turn(ctx)
            assert result.action == MiddlewareAction.INJECT_MESSAGE
            assert result.message == REMINDER

            current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
            result = await middleware.before_turn(ctx)
            assert result.action == MiddlewareAction.INJECT_MESSAGE
            assert result.message == EXIT_MSG

    @pytest.mark.asyncio
    async def test_exit_to_non_default_agent(self, ctx: ConversationContext) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        await middleware.before_turn(ctx)

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.AUTO_APPROVE]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == EXIT_MSG

    @pytest.mark.asyncio
    async def test_switching_between_non_target_agents(
        self, ctx: ConversationContext
    ) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        middleware = _build_middleware(lambda: current_profile)

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.AUTO_APPROVE]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.ACCEPT_EDITS]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_non_target_to_target_entry(self, ctx: ConversationContext) -> None:
        """Starting in a non-target agent then entering target should inject reminder."""
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.AUTO_APPROVE]
        middleware = _build_middleware(lambda: current_profile)

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == REMINDER

    @pytest.mark.asyncio
    async def test_reset_while_inactive_after_exit(
        self, ctx: ConversationContext
    ) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        await middleware.before_turn(ctx)
        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        await middleware.before_turn(ctx)

        middleware.reset()

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_reset_while_inactive_then_reenter(
        self, ctx: ConversationContext
    ) -> None:
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        await middleware.before_turn(ctx)
        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        await middleware.before_turn(ctx)

        middleware.reset()

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == REMINDER

    @pytest.mark.asyncio
    async def test_reset_with_compact_reason(self, ctx: ConversationContext) -> None:
        middleware = _build_middleware(lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN])
        await middleware.before_turn(ctx)

        middleware.reset(ResetReason.COMPACT)

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == REMINDER

    @pytest.mark.asyncio
    async def test_entry_then_continuation_then_exit_then_continuation(
        self, ctx: ConversationContext
    ) -> None:
        """Each call sees one transition at a time."""
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        middleware = _build_middleware(lambda: current_profile)

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == REMINDER

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.DEFAULT]
        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == EXIT_MSG

        result = await middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE


PLAN_REMINDER_SNIPPET = "Plan mode is active"


class TestMiddlewarePipelineWithReadOnlyAgent:
    @pytest.mark.asyncio
    async def test_pipeline_includes_injection(self, ctx: ConversationContext) -> None:
        plan_reminder = make_plan_agent_reminder("/tmp/test-plan.md")
        pipeline = MiddlewarePipeline()
        pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: BUILTIN_AGENTS[BuiltinAgentName.PLAN],
                BuiltinAgentName.PLAN,
                plan_reminder,
                plan_agent_exit,
            )
        )

        result = await pipeline.run_before_turn(ctx)

        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert PLAN_REMINDER_SNIPPET in (result.message or "")

    @pytest.mark.asyncio
    async def test_pipeline_skips_injection_when_not_target_agent(
        self, ctx: ConversationContext
    ) -> None:
        plan_reminder = make_plan_agent_reminder("/tmp/test-plan.md")
        pipeline = MiddlewarePipeline()
        pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: BUILTIN_AGENTS[BuiltinAgentName.DEFAULT],
                BuiltinAgentName.PLAN,
                plan_reminder,
                plan_agent_exit,
            )
        )

        result = await pipeline.run_before_turn(ctx)

        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_direct_plan_to_chat_transition_delivers_both_messages(
        self, ctx: ConversationContext
    ) -> None:
        plan_reminder = make_plan_agent_reminder("/tmp/test-plan.md")
        current_profile: AgentProfile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        pipeline = MiddlewarePipeline()
        pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: current_profile,
                BuiltinAgentName.PLAN,
                plan_reminder,
                plan_agent_exit,
            )
        )
        pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: current_profile,
                BuiltinAgentName.CHAT,
                chat_agent_reminder,
                chat_agent_exit,
            )
        )

        result = await pipeline.run_before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert PLAN_REMINDER_SNIPPET in (result.message or "")

        current_profile = CHAT
        result = await pipeline.run_before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert plan_agent_exit() in (result.message or "")
        assert chat_agent_reminder() in (result.message or "")

        current_profile = BUILTIN_AGENTS[BuiltinAgentName.PLAN]
        result = await pipeline.run_before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert chat_agent_exit() in (result.message or "")
        assert PLAN_REMINDER_SNIPPET in (result.message or "")


def _find_plan_middleware(agent) -> ReadOnlyAgentMiddleware:
    return next(
        mw
        for mw in agent.middleware_pipeline.middlewares
        if isinstance(mw, ReadOnlyAgentMiddleware)
        and mw._agent_name == BuiltinAgentName.PLAN
    )


class TestReadOnlyAgentMiddlewareIntegration:
    @pytest.mark.asyncio
    async def test_switch_agent_preserves_middleware_state_for_exit_message(
        self,
    ) -> None:
        config = build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
            enabled_tools=[],
        )
        agent = build_test_agent_loop(config=config, agent_name=BuiltinAgentName.PLAN)

        plan_middleware = _find_plan_middleware(agent)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert PLAN_REMINDER_SNIPPET in (result.message or "")

        await agent.switch_agent(BuiltinAgentName.DEFAULT)

        plan_middleware_after = _find_plan_middleware(agent)
        assert plan_middleware is plan_middleware_after

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware_after.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == plan_agent_exit()

    @pytest.mark.asyncio
    async def test_switch_agent_allows_reinjection_on_reentry(self) -> None:
        config = build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
            enabled_tools=[],
        )
        agent = build_test_agent_loop(config=config, agent_name=BuiltinAgentName.PLAN)

        plan_middleware = _find_plan_middleware(agent)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        await plan_middleware.before_turn(ctx)

        await agent.switch_agent(BuiltinAgentName.DEFAULT)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware.before_turn(ctx)
        assert result.message == plan_agent_exit()

        await agent.switch_agent(BuiltinAgentName.PLAN)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert PLAN_REMINDER_SNIPPET in (result.message or "")

    @pytest.mark.asyncio
    async def test_switch_plan_to_auto_approve_fires_exit(self) -> None:
        config = build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
            enabled_tools=[],
        )
        agent = build_test_agent_loop(config=config, agent_name=BuiltinAgentName.PLAN)

        plan_middleware = _find_plan_middleware(agent)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        await plan_middleware.before_turn(ctx)  # enter plan

        await agent.switch_agent(BuiltinAgentName.AUTO_APPROVE)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message == plan_agent_exit()

    @pytest.mark.asyncio
    async def test_switch_between_non_plan_agents_no_injection(self) -> None:
        config = build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
            enabled_tools=[],
        )
        agent = build_test_agent_loop(
            config=config, agent_name=BuiltinAgentName.DEFAULT
        )

        plan_middleware = _find_plan_middleware(agent)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

        await agent.switch_agent(BuiltinAgentName.AUTO_APPROVE)

        ctx = ConversationContext(
            messages=agent.messages, stats=agent.stats, config=agent.config
        )
        result = await plan_middleware.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_full_lifecycle_plan_default_plan_default(self) -> None:
        """Integration test for a full plan -> default -> plan -> default cycle."""
        config = build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
            include_model_info=False,
            include_commit_signature=False,
            enabled_tools=[],
        )
        agent = build_test_agent_loop(config=config, agent_name=BuiltinAgentName.PLAN)

        plan_middleware = _find_plan_middleware(agent)

        def _ctx():
            return ConversationContext(
                messages=agent.messages, stats=agent.stats, config=agent.config
            )

        # 1. Enter plan: inject reminder
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.INJECT_MESSAGE
        assert PLAN_REMINDER_SNIPPET in (r.message or "")

        # 2. Stay in plan: no injection
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.CONTINUE

        # 3. Switch to default: inject exit
        await agent.switch_agent(BuiltinAgentName.DEFAULT)
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.INJECT_MESSAGE
        assert r.message == plan_agent_exit()

        # 4. Stay in default: no injection
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.CONTINUE

        # 5. Switch back to plan: inject reminder again
        await agent.switch_agent(BuiltinAgentName.PLAN)
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.INJECT_MESSAGE
        assert PLAN_REMINDER_SNIPPET in (r.message or "")

        # 6. Stay in plan: no injection
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.CONTINUE

        # 7. Switch to default again: inject exit
        await agent.switch_agent(BuiltinAgentName.DEFAULT)
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.INJECT_MESSAGE
        assert r.message == plan_agent_exit()

        # 8. Stay in default: no injection
        r = await plan_middleware.before_turn(_ctx())
        assert r.action == MiddlewareAction.CONTINUE


# --- Loop breakers -----------------------------------------------------------


def _assistant_with_call(name: str, arguments: str, call_id: str = "c") -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content="",
        tool_calls=[
            ToolCall(id=call_id, index=0, function=FunctionCall(name=name, arguments=arguments))
        ],
    )


def _tool_response(call_id: str = "c") -> LLMMessage:
    return LLMMessage(role=Role.tool, tool_call_id=call_id, name="todo", content="ok")


def _conversation(*messages: LLMMessage) -> ConversationList:
    conv = ConversationList()
    for m in messages:
        conv.add(m)
    return conv


def _ctx(vibe_config: VibeConfig, conv: ConversationList, turn_id: str | None = "t1") -> ConversationContext:
    return ConversationContext(
        messages=conv, stats=AgentStats(), config=vibe_config, turn_id=turn_id
    )


def _budget_ctx(
    budget: int, conv: ConversationList | None = None, turn_id: str | None = "t1"
) -> ConversationContext:
    return _ctx(
        build_test_vibe_config(max_llm_calls_per_turn=budget),
        conv if conv is not None else _conversation(),
        turn_id,
    )


class TestStepBudgetMiddleware:
    @pytest.mark.asyncio
    async def test_first_n_calls_continue(self) -> None:
        mw = StepBudgetMiddleware()
        ctx = _budget_ctx(3)
        for _ in range(3):
            assert (await mw.before_turn(ctx)).action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_call_after_budget_requests_writeup_with_tools_suspended(self) -> None:
        mw = StepBudgetMiddleware()
        ctx = _budget_ctx(2)
        await mw.before_turn(ctx)
        await mw.before_turn(ctx)

        result = await mw.before_turn(ctx)

        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.message is not None
        assert "Step budget spent" in result.message
        assert "2 tool-calling steps" in result.message
        assert result.metadata[SUSPEND_TOOLS_METADATA_KEY] is True

    @pytest.mark.asyncio
    async def test_call_after_writeup_request_stops(self) -> None:
        mw = StepBudgetMiddleware()
        ctx = _budget_ctx(1)
        await mw.before_turn(ctx)
        assert (await mw.before_turn(ctx)).action == MiddlewareAction.INJECT_MESSAGE

        result = await mw.before_turn(ctx)

        assert result.action == MiddlewareAction.STOP
        assert result.reason is not None
        assert "Step budget spent" in result.reason

    @pytest.mark.asyncio
    async def test_new_turn_id_resets_count(self) -> None:
        mw = StepBudgetMiddleware()
        conv = _conversation()
        await mw.before_turn(_budget_ctx(1, conv, turn_id="t1"))
        assert (await mw.before_turn(_budget_ctx(1, conv, turn_id="t1"))).action == MiddlewareAction.INJECT_MESSAGE

        result = await mw.before_turn(_budget_ctx(1, conv, turn_id="t2"))

        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_compact_reset_keeps_count_stop_reset_clears_it(self) -> None:
        mw = StepBudgetMiddleware()
        ctx = _budget_ctx(1)
        await mw.before_turn(ctx)

        mw.reset(ResetReason.COMPACT)
        assert (await mw.before_turn(ctx)).action == MiddlewareAction.INJECT_MESSAGE

        mw.reset(ResetReason.STOP)
        assert (await mw.before_turn(ctx)).action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_zero_disables_budget(self) -> None:
        mw = StepBudgetMiddleware()
        ctx = _budget_ctx(0)
        for _ in range(50):
            assert (await mw.before_turn(ctx)).action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_limit_is_read_live_from_config(self) -> None:
        # /llm-calls-per-turn mid-turn: a raised limit lets the turn continue.
        mw = StepBudgetMiddleware()
        conv = _conversation()
        await mw.before_turn(_budget_ctx(1, conv))

        result = await mw.before_turn(_budget_ctx(5, conv))

        assert result.action == MiddlewareAction.CONTINUE


class TestRepeatedToolCallMiddleware:
    @pytest.mark.asyncio
    async def test_three_identical_calls_in_a_row_stop(self, vibe_config: VibeConfig) -> None:
        conv = _conversation(
            LLMMessage(role=Role.user, content="go"),
            _assistant_with_call("todo", '{"action": "read"}', "c1"),
            _tool_response("c1"),
            _assistant_with_call("todo", '{"action":"read"}', "c2"),
            _tool_response("c2"),
            _assistant_with_call("todo", '{ "action" : "read" }', "c3"),
            _tool_response("c3"),
        )

        result = await RepeatedToolCallMiddleware().before_turn(_ctx(vibe_config, conv))

        assert result.action == MiddlewareAction.STOP
        assert "identical `todo` call in 3 consecutive messages" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_two_identical_calls_continue(self, vibe_config: VibeConfig) -> None:
        conv = _conversation(
            LLMMessage(role=Role.user, content="go"),
            _assistant_with_call("todo", '{"action": "read"}', "c1"),
            _tool_response("c1"),
            _assistant_with_call("todo", '{"action": "read"}', "c2"),
            _tool_response("c2"),
        )

        result = await RepeatedToolCallMiddleware().before_turn(_ctx(vibe_config, conv))

        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_different_arguments_continue(self, vibe_config: VibeConfig) -> None:
        conv = _conversation(
            LLMMessage(role=Role.user, content="go"),
            _assistant_with_call("todo", '{"action": "read"}', "c1"),
            _tool_response("c1"),
            _assistant_with_call("todo", '{"action": "write", "todos": []}', "c2"),
            _tool_response("c2"),
            _assistant_with_call("todo", '{"action": "read"}', "c3"),
            _tool_response("c3"),
        )

        result = await RepeatedToolCallMiddleware().before_turn(_ctx(vibe_config, conv))

        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_real_user_message_breaks_the_streak(self, vibe_config: VibeConfig) -> None:
        conv = _conversation(
            LLMMessage(role=Role.user, content="go"),
            _assistant_with_call("todo", '{"action": "read"}', "c1"),
            _tool_response("c1"),
            _assistant_with_call("todo", '{"action": "read"}', "c2"),
            _tool_response("c2"),
            LLMMessage(role=Role.user, content="again please"),
            _assistant_with_call("todo", '{"action": "read"}', "c3"),
            _tool_response("c3"),
        )

        result = await RepeatedToolCallMiddleware().before_turn(_ctx(vibe_config, conv))

        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_injected_user_message_does_not_break_the_streak(
        self, vibe_config: VibeConfig
    ) -> None:
        conv = _conversation(
            LLMMessage(role=Role.user, content="go"),
            _assistant_with_call("todo", '{"action": "read"}', "c1"),
            _tool_response("c1"),
            LLMMessage(role=Role.user, content="reminder", injected=True),
            _assistant_with_call("todo", '{"action": "read"}', "c2"),
            _tool_response("c2"),
            _assistant_with_call("todo", '{"action": "read"}', "c3"),
            _tool_response("c3"),
        )

        result = await RepeatedToolCallMiddleware().before_turn(_ctx(vibe_config, conv))

        assert result.action == MiddlewareAction.STOP


class TestPipelineMetadataMerge:
    @pytest.mark.asyncio
    async def test_injected_results_keep_their_metadata(self) -> None:
        mw = StepBudgetMiddleware()
        pipeline = MiddlewarePipeline().add(mw)
        ctx = _budget_ctx(1)
        await pipeline.run_before_turn(ctx)

        result = await pipeline.run_before_turn(ctx)

        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert result.metadata.get(SUSPEND_TOOLS_METADATA_KEY) is True
