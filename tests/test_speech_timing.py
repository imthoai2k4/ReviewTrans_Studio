from __future__ import annotations

import pytest
import json

from reviewtrans.core.models import Segment
from reviewtrans.core.pipeline import resources
from reviewtrans.core.config import ProviderProfile
from reviewtrans.core.providers.translate.base import LLMTranslator, TranslateJob


def test_translation_duration_budget_follows_lines_when_batch_splits():
    calls = []

    class Translator(LLMTranslator):
        def _call(self, system, prompt, key):
            items = json.loads(prompt.split("Items (JSON):\n")[1].split("\n\nReturn")[0])
            calls.append(items)
            if len(items) > 1:
                return "invalid"
            return json.dumps([{"id": 1, "text": "Translated"}])

    translator = Translator(ProviderProfile(kind="openai"))
    job = TranslateJob(durations=[0.5, 1, 1.5, 2])
    assert translator.translate(["a", "b", "c", "d"], job) == ["Translated"] * 4
    singles = [items[0] for items in calls if len(items) == 1]
    assert [(item["text"], item["duration_seconds"]) for item in singles] == [("a", 0.5), ("b", 1), ("c", 1.5), ("d", 2)]
