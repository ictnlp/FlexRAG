import asyncio
import logging

import pytest

from flexrag.assistants import AssistantBase, AssistantResult
from flexrag.common import ChatMessages, ChatTurn
from flexrag.datasets.core import MultiSessionQASample, QASample
from flexrag.tasks.multisession_qa_base import (
    MultiSessionQATask,
    MultiSessionQATaskConfig,
)
from flexrag.tasks.open_qa_base import OpenQATask, OpenQATaskConfig


class _Evaluator:
    def evaluate(self, **kwargs):
        self.responses = kwargs["responses"]
        return {}, {}


class _Assistant(AssistantBase):
    def __init__(self, events):
        super().__init__()
        self.events, self.active, self.maximum = events, 0, 0

    async def _start_episode(self):
        self.events.append("start")

    async def _finish_episode(self):
        self.events.append("finish")

    async def _add_histories(self, histories):
        self.events.append(histories[0].metadata["session_id"])

    async def _answer(self, messages, *, retrieve):
        del retrieve
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        text = messages[-1].text_content
        return AssistantResult(response=ChatTurn(role="assistant", content=f"a:{text}"))


class _TaskMixin:
    def setup(self):
        pass

    load_dataset = load_evaluator = setup

    async def evaluate(self, assistant, sample):
        return await assistant.answer([{"role": "user", "content": sample.question}])


_OpenTask = type("_OpenTask", (_TaskMixin, OpenQATask), {})
_MultiTask = type("_MultiTask", (_TaskMixin, MultiSessionQATask), {})


def _prepare(task, samples, tmp_path):
    task.testset, task.evaluator = samples, _Evaluator()
    task.logger = logging.getLogger("test.assistant_task")
    task.details_path, task.eval_score_path = tmp_path / "details", tmp_path / "scores"


@pytest.mark.asyncio
async def test_assistant_task_scheduling(tmp_path):
    task = _OpenTask(OpenQATaskConfig())
    _prepare(task, [QASample(question=f"q{i}") for i in range(3)], tmp_path)
    assistant = _Assistant([])

    await task.run(lambda: assistant)
    assert assistant.maximum == 3
    assert task.evaluator.responses == ["a:q0", "a:q1", "a:q2"]

    task = _MultiTask(MultiSessionQATaskConfig())
    samples = []
    for group in ("g1", "g2"):
        history = ChatMessages.from_list([], metadata={"session_id": group})
        samples.extend(
            MultiSessionQASample(
                question=f"{group}-q{i}",
                sessions=[history],
                sessions_id=group,
            )
            for i in range(2)
        )
    _prepare(task, samples, tmp_path)
    events, assistants = [], []

    def factory():
        assistants.append(_Assistant(events))
        return assistants[-1]

    await task.run(factory)
    assert [assistant.maximum for assistant in assistants] == [2, 2]
    assert events == ["start", "g1", "finish", "start", "g2", "finish"]
