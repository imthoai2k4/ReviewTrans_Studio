from unittest.mock import Mock

import pytest

from reviewtrans.core.config import ProviderProfile
from reviewtrans.core.providers import ProviderError
from reviewtrans.core.providers.translate.base import LLMTranslator, TranslateJob


def test_daily_free_quota_stops_without_repeating_requests(monkeypatch):
    translator = LLMTranslator(ProviderProfile(name="API dịch", api_keys=["one", "two"]))
    call = Mock(side_effect=RuntimeError("Error code: 429 - Rate limit exceeded: free-models-per-day"))
    sleep = Mock()
    monkeypatch.setattr("reviewtrans.core.providers.translate.base.time.sleep", sleep)
    with pytest.raises(ProviderError, match="Đã hết hạn mức"):
        translator._with_retry(call, job=TranslateJob())
    call.assert_called_once()
    sleep.assert_not_called()


def test_temporary_rate_limit_can_still_recover(monkeypatch):
    translator = LLMTranslator(ProviderProfile())
    call = Mock(side_effect=[RuntimeError("429 Provider temporarily rate limited"), "ok"])
    monkeypatch.setattr("reviewtrans.core.providers.translate.base.time.sleep", Mock())
    assert translator._with_retry(call) == "ok"
    assert call.call_count == 2
