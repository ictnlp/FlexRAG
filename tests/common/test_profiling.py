import asyncio
import importlib
import json

from flexrag.common import record_span, start_session, trace


def test_trace_is_noop_without_session():
    @trace("test.noop")
    def traced() -> str:
        with record_span("test.noop.manual"):
            return "ok"

    assert traced() == "ok"

    with start_session() as session:
        pass

    assert session.summary(show=False) == {}


def test_trace_records_and_exports(tmp_path):
    @trace("test.sync")
    def sync_func() -> str:
        return "sync"

    @trace("test.async")
    async def async_func() -> str:
        return "async"

    async def run_once():
        with start_session() as session:
            assert sync_func() == "sync"
            with record_span("test.manual"):
                assert await async_func() == "async"
        return session

    session = asyncio.run(run_once())
    summary = session.summary(show=False)

    assert summary["test.sync"]["Calls"] == 1
    assert summary["test.async"]["Calls"] == 1
    assert summary["test.manual"]["Calls"] == 1
    assert summary["test.sync"]["Wall Sum"] >= 0.0
    assert summary["test.async"]["Wall Sum"] >= 0.0
    assert summary["test.manual"]["Wall Sum"] >= 0.0

    trace_path = tmp_path / "trace.json"
    session.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text(encoding="utf-8"))

    assert {event["name"] for event in events} == {
        "test.sync",
        "test.async",
        "test.manual",
    }
    assert all(event["ph"] == "X" for event in events)
    assert all(event["pid"] > 0 for event in events)
    assert all(event["dur"] >= 0 for event in events)


def test_migrated_modules_import():
    for module_name in (
        "flexrag.metrics.generation_metrics",
        "flexrag.metrics.llm_as_a_judge.shortform_correctness",
        "flexrag.processors.text_processors.pipeline",
    ):
        importlib.import_module(module_name)
