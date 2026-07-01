"""Unit tests for lib.chat.off_topic detectors."""

from __future__ import annotations

from lib.chat.off_topic import (
    impossible_request_subject,
    is_impossible_catalog_request,
    is_off_topic_message,
    off_topic_topic,
)


def test_is_off_topic_message_weather() -> None:
    assert is_off_topic_message("What's the weather in Colombo?")
    assert off_topic_topic("What's the weather in Colombo?") == "weather"


def test_is_off_topic_message_shopping_not_off_topic() -> None:
    assert not is_off_topic_message("chocolate gift for my wife in Kandy")


def test_is_impossible_catalog_request_elephant() -> None:
    assert is_impossible_catalog_request("Can you deliver a live elephant to Colombo?")
    assert impossible_request_subject("live elephant to Kandy") == "live elephant"


def test_is_impossible_catalog_request_stuffed_ok() -> None:
    assert not is_impossible_catalog_request("stuffed elephant toy for kids")
