from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reviewtrans.core.pipeline.asr_boundaries import aligned_tokens, rebuild_sentences
from reviewtrans.core.pipeline.funasr import segments_from_funasr


def sentence(raw, times, punctuation="，"):
    return {"raw_text": raw, "text": raw.replace(" ", "") + punctuation,
            "start": times[0][0], "end": times[-1][1], "timestamp": times}


def evidence(left, right, score):
    return {"left_end": left, "right_start": right, "similarity": score}


def test_episode_10_split_response_then_rejoin_reassurance():
    # Actual word timestamps from the supplied episode's original ASR cache.
    data = {"sentence_info": [
        sentence("但 我 们 担 心 到 时 候 拖 你 的 后 腿 放 心", [
            [39570,39750],[39750,39810],[39810,40050],[40050,40110],
            [40110,40710],[40710,40829],[40829,40890],[40890,41130],
            [41130,41249],[41249,41309],[41309,41550],[41550,41610],
            [41610,42330],[43290,43350],[43350,44070]]),
        sentence("不 会 的", [[44070,44130],[44130,44310],[44310,45030]]),
        sentence("真 的 吗", [[45210,45270],[45270,45389],[45389,46110]], "？")],
        "voice_boundaries": [evidence(44070, 44070, .465), evidence(45030,45210,.021)]}
    original = copy.deepcopy(data)
    segments = segments_from_funasr({"result": data})
    assert [(s.start,s.end,s.source) for s in segments] == [
        (39.57,42.33,"但我们担心到时候拖你的后腿"),
        (43.29,45.03,"放心，不会的，"),
        (45.21,46.11,"真的吗？")]
    assert data == original


def test_episode_1_quick_reply_requires_voice_evidence():
    data = {"sentence_info": [sentence("多 谢 道 友 施 以 援 手 道 友 客 气", [
        [66869,66930],[66930,67290],[67290,67349],[67349,67770],
        [67770,67830],[67830,68190],[68190,68250],[68250,68969],
        [69090,69150],[69150,69449],[69449,69509],[69509,70230]])]}
    assert len(rebuild_sentences(data)) == 1  # Short pause alone isn't a turn.
    data["voice_boundaries"] = [evidence(68969,69090,.296)]
    result = rebuild_sentences(data)
    assert [(s["text"], s["start"], s["end"]) for s in result] == [
        ("多谢道友施以援手",66869,68969), ("道友客气，",69090,70230)]


def test_long_silence_splits_even_without_punctuation_or_voice_model():
    data = {"sentence_info": [sentence("你 找 我 什 么", [
        [30880,31119],[31119,31180],[31180,32050],[34350,34710],[34710,35175]])]}
    assert [s["text"] for s in rebuild_sentences(data)] == ["你找我", "什么，"]


@pytest.mark.parametrize("score,gap,punc,expected", [
    (.8,0,"，",1), (.8,180,"，",1), (.8,181,"，",2),
    (.2,0,"，",2), (.4,0,"，",2), (.8,0,"。",2),
    (.8,0,"？",2), (None,0,"，",2), (float("nan"),0,"，",2)])
def test_merge_requires_positive_voice_short_pause_and_weak_punctuation(score,gap,punc,expected):
    data = {"sentence_info": [sentence("你 好", [[0,300],[300,800]],punc),
                              sentence("世 界", [[800+gap,1100+gap],[1100+gap,1600+gap]])],
            "voice_boundaries": [evidence(800,800+gap,score)]}
    assert len(rebuild_sentences(data)) == expected


@pytest.mark.parametrize("change", [
    {"raw_text": "错 词"}, {"raw_text": ""}, {"timestamp": []},
    {"timestamp": [[0, 200], [float('nan'), 1200]]},
    {"timestamp": [[0, 200], [1100, 1300]]},
])
def test_unaligned_tokens_remain_unchanged(change):
    item = sentence("你 好", [[0,200],[1100,1200]])
    item.update(change)
    assert aligned_tokens(item) is None
    assert rebuild_sentences({"sentence_info": [item]}) == [item]


def test_mixed_surface_tokens_quotes_and_punctuation_are_preserved():
    item = {"raw_text": "你 好 Hello world", "text": '“你好，Hello world!”',
            "start": 0, "end": 2400,
            "timestamp": [[0,200],[200,400],[1500,1800],[1800,2400]]}
    chunks = rebuild_sentences({"sentence_info": [item]})
    assert [s["text"] for s in chunks] == ['“你好，','Hello world!”']
    assert ''.join(s['text'] for s in chunks) == item['text']


def test_surface_form_raw_text_supported():
    item = sentence("你 好 世 界", [[0,200],[200,400],[1500,1800],[1800,2400]])
    item["raw_text"] = "你好世界"
    assert len(rebuild_sentences({"sentence_info": [item]})) == 2


def test_merge_has_duration_limit():
    data = {"sentence_info": [sentence("你 好", [[0,300],[300,4000]]),
                              sentence("世 界", [[4000,5000],[5000,8000]])],
            "voice_boundaries": [evidence(4000,4000,.9)]}
    assert len(rebuild_sentences(data)) == 2


def test_worker_compares_real_audio_windows_without_crossing_pauses(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("worker", Path(__file__).parents[1]/"scripts/funasr_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    model = Mock()
    model.generate.return_value = [{"spk_embedding": "embedding"}]
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules,"funasr",SimpleNamespace(AutoModel=factory))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(
        cosine_similarity=lambda *_: SimpleNamespace(item=lambda: .2)))))
    samples = list(range(40000))
    result = {"sentence_info": [sentence("你 好 世 界", [[0,400],[400,800],[920,1320],[1320,1720]])]}
    assert worker.voice_boundaries(result,samples,tmp_path,"cpu",2) == [evidence(800,920,.2)]
    assert model.generate.call_args_list[0].kwargs["input"] == samples[:12800]
    assert model.generate.call_args_list[1].kwargs["input"] == samples[14720:27520]
    assert factory.call_args.kwargs["trust_remote_code"] is False


def test_worker_does_not_classify_isolated_syllable(monkeypatch,tmp_path):
    spec = importlib.util.spec_from_file_location("worker", Path(__file__).parents[1]/"scripts/funasr_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    factory = Mock()
    monkeypatch.setitem(sys.modules,"funasr",SimpleNamespace(AutoModel=factory))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    result = {"sentence_info": [sentence("你 已 经 突", [[0,719],[1320,1380],[1380,1700],[1700,1800]])]}
    assert worker.voice_boundaries(result,list(range(30000)),tmp_path,"cpu",2) == []
    factory.return_value.generate.assert_not_called()


def test_all_four_models_and_camplus_weights_are_required(tmp_path, monkeypatch):
    from reviewtrans.core.pipeline import funasr
    monkeypatch.setattr(funasr, "funasr_models_dir", lambda: tmp_path)
    for name in funasr.MODEL_NAMES:
        folder = tmp_path / name
        folder.mkdir()
        for filename in (".reviewtrans-complete", "model.pt", "config.yaml", "configuration.json",
                         "tokens.json", "seg_dict", "am.mvn"):
            (folder / filename).write_text("test", "utf-8")
    assert not funasr.models_ready()  # model.pt is not CAM++'s actual weight file.
    (tmp_path / "campplus/campplus_cn_common.bin").write_bytes(b"weights")
    assert funasr.models_ready()


def test_downloader_includes_camplus_binary_and_completion_marker(tmp_path,monkeypatch):
    spec = importlib.util.spec_from_file_location("worker", Path(__file__).parents[1]/"scripts/funasr_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    def download(**kwargs):
        folder = Path(kwargs["local_dir"])
        folder.mkdir()
        for filename in ("model.pt", "config.yaml", "configuration.json", "tokens.json", "seg_dict", "am.mvn"):
            (folder/filename).write_text("test","utf-8")
        if kwargs["repo_id"] == "funasr/campplus":
            assert "*.bin" in kwargs["allow_patterns"]
            (folder/"campplus_cn_common.bin").write_bytes(b"weights")
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(snapshot_download=download))
    worker.download_models(tmp_path)
    assert (tmp_path/"campplus/.reviewtrans-complete").read_text("utf-8") == "1\n"
