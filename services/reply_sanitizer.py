"""Remove private impression annotations before rendering and streaming delivery."""

from __future__ import annotations

import re

_OPEN = re.compile(
    r"<<\s*impression\s*[:：]|[\[【［]\s*impression\s*[:：]|"
    r"<\s*impression\s*(?:[:：]|>)",
    re.IGNORECASE,
)
_TAIL = re.compile(r"(?:<<|<|\[|【|［)\s*[a-z]*\s*[:：]?\s*$", re.IGNORECASE)
_OPENERS = (
    "<<impression:",
    "[impression:",
    "【impression:",
    "［impression:",
    "<impression>",
    "<impression:",
)


class ImpressionFilter:
    """Incremental parser that holds only an unfinished tag or opening prefix."""

    def __init__(self):
        self.pending = ""
        self.closer = None
        self.body = ""
        self.marks: list[str] = []

    def feed(self, text: str) -> str:
        """Filter a fragment without exposing prefixes split across chunks.

        Args:
            text: The next text fragment, in delivery order.

        Returns:
            Text that can already be delivered to the user.
        """
        remaining = self.pending + text
        self.pending = ""
        visible = []
        while remaining:
            if self.closer is not None:
                match = self.closer.search(remaining)
                if match is None:
                    # XML closing tags can span chunks. Keep a bounded suffix;
                    # the private body is capped independently of input length.
                    self.body = (self.body + remaining[:-32])[:4096]
                    self.pending = remaining[-32:]
                    break
                self.body = (self.body + remaining[: match.start()])[:4096]
                self.marks.append(self.body.strip())
                self.body = ""
                self.closer = None
                remaining = remaining[match.end() :].lstrip()
                continue
            match = _OPEN.search(remaining)
            if match is None:
                tail = _TAIL.search(remaining)
                if tail:
                    prefix = "".join(tail.group().split()).casefold().replace("：", ":")
                    if any(opener.startswith(prefix) for opener in _OPENERS):
                        visible.append(remaining[: tail.start()])
                        self.pending = tail.group()
                        break
                visible.append(remaining)
                break
            visible.append(remaining[: match.start()].rstrip())
            opening = match.group()
            if opening.startswith("<<"):
                closing = r">>"
            elif opening[0] in "[【［":
                closing = {"[": r"\]", "【": "】", "［": "］"}[opening[0]]
            else:
                closing = r"</\s*impression\s*>" if opening.endswith(">") else ">"
            self.closer = re.compile(closing, re.IGNORECASE)
            remaining = remaining[match.end() :]
        return "".join(visible)

    def finish(self) -> str:
        """Discard an unfinished private annotation and flush ordinary text.

        Returns:
            A trailing literal prefix that is not an impression annotation.
        """
        tail = self.pending
        self.pending = ""
        if self.closer is not None or "impr" in tail.casefold():
            self.closer = None
            self.body = ""
            return ""
        return tail


def strip_impression_marks(text: str) -> tuple[str, list[str]]:
    """Strip complete and unfinished annotations from a full response.

    Args:
        text: The model response or stored reply.

    Returns:
        Cleaned text and the extracted private annotations.
    """
    parser = ImpressionFilter()
    cleaned = parser.feed(text) + parser.finish()
    return cleaned.rstrip() if _OPEN.search(text) else cleaned, parser.marks


def sanitize_chain(chain: list) -> list[str]:
    """Clean text runs and forwarded nodes while preserving media order.

    Args:
        chain: Message components to update in place.

    Returns:
        Private annotations extracted from the message chain.
    """
    from astrbot.core.message.components import Node, Nodes, Plain

    marks = []
    run = []
    for comp in [*chain, None]:
        if isinstance(comp, Plain):
            run.append(comp)
            continue
        if run:
            original = "".join(item.text or "" for item in run)
            cleaned, extracted = strip_impression_marks(original)
            if cleaned != original:
                run[0].text = cleaned
                for item in run[1:]:
                    item.text = ""
            marks.extend(extracted)
            run = []
        if isinstance(comp, Node):
            marks.extend(sanitize_chain(comp.content))
        elif isinstance(comp, Nodes):
            for node in comp.nodes:
                marks.extend(sanitize_chain(node.content))
    return marks


async def record_impressions(plugin, event, marks: list[str]) -> None:
    """Persist each private annotation once through the serialized writer.

    Args:
        plugin: Plugin instance owning the database and write gateway.
        event: Event carrying the canonical speaker scope.
        marks: Extracted annotations, including those from the response hook.
    """
    if not marks:
        return
    try:
        from ..domain.scope import RuntimeScope
        from .impression_timeline import parse_impression_mark, persist_unsettled_trace
    except ImportError:  # Focused tests import the services package directly.
        from domain.scope import RuntimeScope
        from services.impression_timeline import (
            parse_impression_mark,
            persist_unsettled_trace,
        )
    from astrbot.api import logger

    scope = getattr(event, "_wave_memory_runtime_scope", None)
    if not isinstance(scope, RuntimeScope):
        resolver = getattr(plugin, "scope_resolver", None)
        if resolver is not None:
            try:
                scope = resolver.resolve_event(event).scope
            except Exception as exc:
                logger.debug("[WaveMemory] impression scope resolution failed: %s", exc)
                return
    if (
        not isinstance(scope, RuntimeScope)
        or scope.visibility != "group"
        or scope.session is None
    ):
        return
    prefix = f"{scope.session.platform_id}:user:"
    principal = scope.subject_principal_id or ""
    user_id = principal[len(prefix) :] if principal.startswith(prefix) else ""
    if not user_id or user_id == "bot":
        return
    recorded = set(getattr(event, "_wave_memory_recorded_impressions", set()))
    pending = []
    for mark in marks:
        body, impact = parse_impression_mark(mark)
        if body and (body, impact) not in recorded:
            recorded.add((body, impact))
            pending.append((body, impact))
    if not pending:
        return

    def persist(connection):
        for body, impact in pending:
            persist_unsettled_trace(
                plugin.db,
                bot_id=scope.bot_id,
                user_id=user_id,
                group_id=scope.session.conversation_id,
                text=body,
                impact=impact,
                connection=connection,
            )
        connection.execute(
            """INSERT INTO user_profiles(user_id, group_id, bot_id, metadata, interaction_count, last_seen)
               VALUES (?, ?, ?, '{}', 1, strftime('%s','now'))
               ON CONFLICT(user_id, group_id, bot_id) DO NOTHING""",
            (user_id, scope.session.conversation_id, scope.bot_id),
        )

    try:
        coordinator = getattr(
            getattr(plugin, "write_gateway", None), "coordinator", None
        )
        if coordinator is None:
            logger.warning("[WaveMemory] impression persistence skipped: writer unavailable")
            return
        await coordinator.transaction(persist)
    except Exception as exc:
        logger.warning("[WaveMemory] impression persistence failed: %s", exc)
    else:
        event._wave_memory_recorded_impressions = recorded


async def sanitize_stream(stream, plugin, event):
    """Filter annotations across streaming chunks without buffering the reply.

    Args:
        stream: Original asynchronous stream of MessageChain objects.
        plugin: Plugin owning the impression store.
        event: Current reply event.

    Yields:
        Sanitized chunks with their original stream metadata.
    """
    from astrbot.core.message.components import Node, Nodes, Plain
    from astrbot.core.message.message_event_result import MessageChain

    parser = ImpressionFilter()
    try:
        async for chunk in stream:
            # Reasoning frames have a separate renderer; do not carry their
            # trailing text into a public response chunk.
            if chunk.type == "reasoning":
                sanitize_chain(chunk.chain)
                yield chunk
                continue
            for comp in chunk.chain:
                if isinstance(comp, Plain):
                    comp.text = parser.feed(comp.text or "")
                elif isinstance(comp, Node):
                    parser.marks.extend(sanitize_chain(comp.content))
                elif isinstance(comp, Nodes):
                    for node in comp.nodes:
                        parser.marks.extend(sanitize_chain(node.content))
            if chunk.type not in {None, "plain"} or any(
                not isinstance(comp, Plain) or comp.text for comp in chunk.chain
            ):
                yield chunk
        tail = parser.finish()
        if tail:
            yield MessageChain(chain=[Plain(tail)])
    finally:
        await record_impressions(plugin, event, parser.marks)
