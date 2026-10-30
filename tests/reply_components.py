"""Scoped message-component doubles for reply tests without the AstrBot SDK."""

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from unittest.mock import patch

import pytest


@dataclass
class Plain:
    text: str


@dataclass
class Image:
    file: str

    @classmethod
    def fromURL(cls, url):
        return cls(url)


@dataclass
class Node:
    name: str
    uin: str
    content: list


@dataclass
class Nodes:
    nodes: list


@dataclass
class MessageChain:
    chain: list = field(default_factory=list)
    type: str | None = None


@contextmanager
def component_modules():
    """Install message doubles only for the duration of one reply test.

    Yields:
        A scope in which production imports resolve to the test components.
    """
    components = ModuleType("astrbot.core.message.components")
    components.Plain = Plain
    components.Image = Image
    components.Node = Node
    components.Nodes = Nodes
    messages = ModuleType("astrbot.core.message.message_event_result")
    messages.MessageChain = MessageChain
    modules = {components.__name__: components, messages.__name__: messages}
    if "astrbot.api" not in sys.modules:
        api = ModuleType("astrbot.api")
        api.logger = logging.getLogger("astrbot-test")
        modules[api.__name__] = api
    with patch.dict("sys.modules", modules):
        yield


@pytest.fixture
def reply_components():
    with component_modules():
        yield
