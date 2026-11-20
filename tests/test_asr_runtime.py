"""Regression coverage for replacing the retired ASR backend."""
import json
import pytest
from reviewtrans.core.config import SettingsStore, AppSettings, ASR_ENGINES
from reviewtrans.core.models import Project, VideoDoc, from_dict, to_dict
from reviewtrans.core.pipeline import RunContext, asr
from reviewtrans.core.store import ProjectStore
from reviewtrans.core.hardware import ASR_DEVICES


def test_settings_migrate_without_losing_other_preferences(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(dict(settings_version=4, default_asr_engine='whisper',
        default_whisper_model='large-v3', whisper_threads=6, asr_device='vulkan',
        default_target_language='en')), encoding='utf-8')
    settings = SettingsStore(path).settings
    assert settings.default_asr_engine == 'paraformer'
    assert settings.asr_threads == 6 and settings.asr_device == 'auto'
    assert settings.default_target_language == 'en'
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert not any('whisper' in key for key in saved)
    assert [key for key,_ in ASR_ENGINES] == ['paraformer']
    assert 'vulkan' not in [key for key,_ in ASR_DEVICES]


@pytest.mark.parametrize('cls', [Project, VideoDoc])
def test_old_project_and_video_config_migrates(cls):
    obj = from_dict(cls, dict(name='keep', whisper_model='large-v3', asr_engine='whisper'))
    assert obj.name == 'keep' and obj.asr_engine == 'paraformer'
    assert 'whisper_model' not in to_dict(obj)


def test_other_languages_require_srt(tmp_path):
    ctx = RunContext(ProjectStore(tmp_path), Project(source_language='vi'), AppSettings())
    with pytest.raises(ValueError, match='SRT'):
        asr.resolve_asr_engine(ctx, VideoDoc())
