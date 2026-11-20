from __future__ import annotations

import tempfile
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtMultimedia, QtWidgets

from ...core.config import THINKING_OPTIONS, TRANSLATE_KINDS, TTS_KINDS, ProviderProfile
from ...core.paths import temp_dir
from ...core.models import clone, new_id
from ...core.providers.translate import SUGGESTED_MODELS, create_translator
from ...core.providers.translate.thinking import selectable_modes, supported_modes
from ...core.providers.tts import create_tts
from ..jobs import run_background
from ..state import AppState
from ..widgets.common import ComboBox, confirm, form_layout, hint, push_button, scroll_wrap, title_label

# gợi ý theo loại: (placeholder base_url, placeholder model, các option thêm)
KIND_HINTS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "Mọi API tương thích OpenAI Chat Completions: OpenAI, DeepSeek, OpenRouter, Groq, LM Studio, Ollama (…/v1)."),
    "gemini": ("(để trống = Google AI Studio)", "gemini-2.5-flash", "API kiểu Gemini (google-genai). Có thể điền base URL của proxy tương thích."),
    "anthropic": ("(để trống = api.anthropic.com)", "claude-opus-5", "Claude qua SDK chính thức. Base URL để trống sẽ bật fallback phía server khi model từ chối."),
    "vbee": ("https://vbee.vn/api/v1", "", "Cần App ID + token (điền token vào ô API key)."),
    "openai_speech": ("https://api.openai.com/v1", "trống = tự lấy từ {base}/models",
                      "Mọi server có POST {base}/audio/speech kiểu OpenAI. Giọng/model lấy từ GET {base}/voices và /models nếu server có."),
    "legacy_custom": ("https://…", "", "API cũ: POST form {text, language} → {code:0, data:url}."),
}

OPTION_FIELDS = {
    "openai": [("temperature", "Temperature", "0.3")],
    "gemini": [("temperature", "Temperature", "0.3")],
    "anthropic": [("temperature", "Temperature", "0.3"), ("fallbacks", "Fallback khi từ chối (on/off)", "on")],
    "vbee": [("app_id", "App ID", ""), ("voice", "Voice code", ""), ("max_polls", "Số lần chờ tối đa", "30"),
             ("callback_url", "Callback URL", "https://example.com/callback"), ("concurrency", "Số luồng", "0")],
    "openai_speech": [("voice", "Giọng mặc định", "trống = giọng đầu tiên của server"), ("voices", "Thêm giọng (phẩy)", ""),
                      ("format", "Định dạng", "wav"), ("instructions", "Chỉ dẫn giọng đọc", ""), ("concurrency", "Số luồng", "0")],
    "legacy_custom": [("language", "Language", "vi"), ("concurrency", "Số luồng", "0")],
}


class ProfileEditor(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, kinds: dict[str, str], parent=None):
        super().__init__(parent)
        self.kinds = kinds
        self.profile: ProviderProfile | None = None
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Cấu hình")
        self.form = form_layout()
        box.setLayout(self.form)
        self.name = QtWidgets.QLineEdit()
        self.kind = ComboBox(list(kinds.items()))
        self.base_url = QtWidgets.QLineEdit()
        self.keys = QtWidgets.QPlainTextEdit()
        self.keys.setPlaceholderText("Mỗi dòng một key — sẽ tự xoay vòng khi gặp giới hạn")
        self.keys.setMaximumHeight(80)
        self.model = ComboBox()
        self.model.setEditable(True)
        self.form.addRow("Tên", self.name)
        self.form.addRow("Loại", self.kind)
        self.form.addRow("Base URL", self.base_url)
        self.form.addRow("API key", self.keys)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model, 1)
        self.fetch_models = QtWidgets.QPushButton("Tải danh sách model")
        self.fetch_models.clicked.connect(self._fetch_models)
        model_row.addWidget(self.fetch_models)
        self.form.addRow("Model", model_row)
        layout.addWidget(box)
        self.options_box = QtWidgets.QGroupBox("Tuỳ chọn")
        self.options_form = form_layout()
        self.options_box.setLayout(self.options_form)
        layout.addWidget(self.options_box)
        self.hint = hint("")
        layout.addWidget(self.hint)
        layout.addStretch()
        self.option_edits: dict[str, QtWidgets.QLineEdit] = {}
        self.thinking_combos: dict[str, ComboBox] = {}
        self.name.editingFinished.connect(self._commit)
        self.base_url.editingFinished.connect(self._commit)
        self.keys.textChanged.connect(self._commit)
        self.model.currentTextChanged.connect(self._commit)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.setEnabled(False)

    def set_profile(self, profile: ProviderProfile | None, kinds: dict[str, str] | None = None) -> None:
        self.profile = profile
        self.setEnabled(profile is not None)
        if profile is None:
            self.hide()
            return
        self.show()
        self._loading = True
        self.fetch_models.setEnabled(True)
        self.fetch_models.setText("Tải danh sách model")
        if kinds is not None:
            self.kinds = kinds
            self.kind.set_items(list(kinds.items()), keep=False)
        self.name.setText(profile.name)
        self.kind.set_value(profile.kind)
        self.base_url.setText(profile.base_url)
        self.keys.setPlainText("\n".join(profile.api_keys))
        self._fill_models(profile.kind)
        self.model.setEditText(profile.model)
        self._build_options(profile)
        self._loading = False

    def _fill_models(self, kind: str) -> None:
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(SUGGESTED_MODELS.get(kind, []))
        self.model.blockSignals(False)

    def _build_options(self, profile: ProviderProfile) -> None:
        while self.options_form.rowCount():
            self.options_form.removeRow(0)
        self.option_edits.clear()
        self.thinking_combos.clear()
        fields = OPTION_FIELDS.get(profile.kind, [])
        self.options_box.setVisible(bool(fields))
        for key, label, default in fields:
            edit = QtWidgets.QLineEdit(str(profile.options.get(key, "")))
            edit.setPlaceholderText(default)
            if key == "concurrency":
                edit.setValidator(QtGui.QIntValidator(0, 16, edit))
                edit.setToolTip("Để trống: theo Cài đặt (mặc định 0). Nhập 0 hoặc 1 để gửi tuần tự; từ 2 để gửi song song.")
            edit.editingFinished.connect(self._commit)
            self.option_edits[key] = edit
            if key == "voice" and profile.kind == "openai_speech":
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(edit, 1)
                pick = QtWidgets.QPushButton("Chọn giọng…")
                pick.clicked.connect(lambda _=False, b=pick: self._pick_voice(b))
                row_layout.addWidget(pick)
                self.options_form.addRow(label, row)
            else:
                self.options_form.addRow(label, edit)
        if profile.kind == "openai":
            for key, label in (
                ("subtitle_thinking", "Thinking khi dịch phụ đề"),
                ("context_thinking", "Thinking khi cập nhật ngữ cảnh"),
            ):
                combo = ComboBox(THINKING_OPTIONS)
                combo.set_value(profile.option(key, ""))
                combo.setToolTip("OpenRouter cho phép chọn khi thiếu thông tin hỗ trợ; mức thực tế phụ thuộc model. Cấu hình được lưu riêng cho từng bước.")
                combo.currentIndexChanged.connect(self._commit)
                self.thinking_combos[key] = combo
                self.options_form.addRow(label, combo)
            self.options_form.addRow(hint(
                "Gợi ý: dịch phụ đề dùng Tắt/Thấp; cập nhật ngữ cảnh thử Vừa, hoặc Bật tự động. "
                "Cao dành cho nội dung phức tạp và có thể chậm, tốn token hơn. "
                "OpenRouter: vẫn chọn được khi danh sách model thiếu thông tin thinking; "
                "API có thể điều chỉnh hoặc từ chối mức không hỗ trợ. "
                "Tắt chỉ bị khóa nếu model được xác nhận bắt buộc thinking. DeepSeek có thể chỉ hỗ trợ Thấp/Cao."
            ))
            self.options_form.addRow(push_button("Áp dụng mức thinking gợi ý", self._recommend_thinking))
            self._update_thinking_choices()
        url_hint, model_hint, text = KIND_HINTS.get(profile.kind, ("", "", ""))
        self.base_url.setPlaceholderText(url_hint)
        self.model.lineEdit().setPlaceholderText(model_hint)
        self.model.setEnabled(profile.kind not in ("vbee", "legacy_custom"))
        self.fetch_models.setVisible(profile.kind in ("openai", "gemini", "anthropic", "openai_speech"))
        self.hint.setText(text)

    def _update_thinking_choices(self) -> None:
        if not self.profile or self.profile.kind != "openai":
            return
        model = self.model.currentText().strip()
        allowed = selectable_modes(self.profile, model)
        confirmed = supported_modes(self.profile, model)
        for combo in self.thinking_combos.values():
            for index in range(combo.count()):
                mode = combo.itemData(index)
                item = combo.model().item(index)
                item.setEnabled(mode in allowed)
                item.setToolTip(
                    "" if mode in confirmed else
                    "Chưa xác nhận mức hỗ trợ; API sẽ xử lý lựa chọn này." if mode in allowed else
                    "Model/API này không cho phép lựa chọn này."
                )

    def _recommend_thinking(self) -> None:
        if not self.profile:
            return
        allowed = supported_modes(self.profile, self.model.currentText().strip())
        for key, preferred in (
            ("subtitle_thinking", ("disabled", "low", "")),
            ("context_thinking", ("medium", "enabled", "")),
        ):
            self.thinking_combos[key].set_value(next(mode for mode in preferred if mode in allowed))

    def _fetch_models(self) -> None:
        if not self.profile:
            return
        self._commit()
        snapshot = clone(self.profile)
        self.fetch_models.setEnabled(False)
        self.fetch_models.setText("Đang tải…")

        def done(models: list[str]) -> None:
            if self.profile is None or (self.profile.id, self.profile.kind) != (snapshot.id, snapshot.kind):
                return
            self.fetch_models.setEnabled(True)
            self.fetch_models.setText("Tải danh sách model")
            if self.profile.base_url != snapshot.base_url:
                return
            current = self.model.currentText()
            self.model.blockSignals(True)
            self.model.clear()
            self.model.addItems(models)
            self.model.setEditText(current)
            self.model.blockSignals(False)
            for key in ("reasoning_catalog", "reasoning_catalog_url"):
                if key in snapshot.options:
                    self.profile.options[key] = snapshot.options[key]
            self._update_thinking_choices()
            self.changed.emit()
            self.model.showPopup()

        def fail(message: str) -> None:
            if self.profile is None or (self.profile.id, self.profile.kind) != (snapshot.id, snapshot.kind):
                return
            self.fetch_models.setEnabled(True)
            self.fetch_models.setText("Tải danh sách model")
            QtWidgets.QMessageBox.critical(self, "Không lấy được danh sách model", message[:500])

        if snapshot.kind == "openai_speech":
            run_background(lambda _p, _s: create_tts(snapshot).remote_models(), done, fail)
        else:
            run_background(lambda _p, _s: create_translator(snapshot).list_models(), done, fail)

    def _pick_voice(self, button: QtWidgets.QPushButton) -> None:
        if not self.profile:
            return
        self._commit()
        snapshot = clone(self.profile)
        voice_edit = self.option_edits.get("voice")
        button.setEnabled(False)
        button.setText("Đang tải…")

        def done(voices: list[tuple[str, str]]) -> None:
            if self.profile is None or self.profile.id != snapshot.id or self.option_edits.get("voice") is not voice_edit:
                return
            button.setEnabled(True)
            button.setText("Chọn giọng…")
            if not voices:
                QtWidgets.QMessageBox.information(self, "Chọn giọng", "Server không trả danh sách giọng.")
                return
            labels = [label for _id, label in voices]
            current = self.option_edits["voice"].text().strip()
            index = next((i for i, (vid, _l) in enumerate(voices) if vid == current), 0)
            label, ok = QtWidgets.QInputDialog.getItem(self, "Chọn giọng", "Giọng:", labels, index, False)
            if ok:
                self.option_edits["voice"].setText(voices[labels.index(label)][0])
                self._commit()

        def fail(message: str) -> None:
            if self.profile is None or self.profile.id != snapshot.id or self.option_edits.get("voice") is not voice_edit:
                return
            button.setEnabled(True)
            button.setText("Chọn giọng…")
            QtWidgets.QMessageBox.critical(self, "Không lấy được danh sách giọng", message[:500])

        run_background(lambda _p, _s: create_tts(snapshot).list_voices("vi"), done, fail)

    def _kind_changed(self, _index: int) -> None:
        if self._loading or not self.profile:
            return
        old_label = self.kinds.get(self.profile.kind, "")
        if self.profile.name in (f"{old_label} mới", old_label):
            self.profile.name = f"{self.kinds.get(self.kind.value(), '')} mới"
            self.name.setText(self.profile.name)
        self._loading = True
        self.profile.kind = self.kind.value()
        self.profile.options.clear()
        self._fill_models(self.profile.kind)
        self._build_options(self.profile)
        self._loading = False
        self._commit()

    def _commit(self, *_args) -> None:
        if self._loading or not self.profile:
            return
        p = self.profile
        p.name = self.name.text().strip() or p.name
        p.base_url = self.base_url.text().strip()
        p.api_keys = [k.strip() for k in self.keys.toPlainText().splitlines() if k.strip()]
        p.model = self.model.currentText().strip()
        self._update_thinking_choices()
        for key, edit in self.option_edits.items():
            value = edit.text().strip()
            if value:
                p.options[key] = value
            else:
                p.options.pop(key, None)
        for key, combo in self.thinking_combos.items():
            value = combo.value()
            if value:
                p.options[key] = value
            else:
                p.options.pop(key, None)
        self.changed.emit()


class ApiProfilesPanel(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.kind = "translate"
        self.kinds = TRANSLATE_KINDS
        self.entries: list[tuple[str, ProviderProfile]] = []
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        left = QtWidgets.QVBoxLayout()
        active_box = QtWidgets.QGroupBox("API đang sử dụng")
        active_form = form_layout()
        active_box.setLayout(active_form)
        self.active_profiles: dict[str, ComboBox] = {}
        for category, label in (("translate", "Dịch"), ("tts", "Lồng tiếng")):
            combo = ComboBox()
            combo.setMinimumContentsLength(16)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.activated.connect(lambda _index, category=category: self._switch_api(category))
            self.active_profiles[category] = combo
            active_form.addRow(label, combo)
        active_form.addRow(hint(
            "Chọn API để chuyển và lưu ngay. Áp dụng khi project/video dùng API mặc định."
        ))
        left.addWidget(active_box)
        left.addWidget(hint("API dịch và lồng tiếng"))
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumWidth(240)
        self.list.setMaximumWidth(380)
        self.list.currentRowChanged.connect(self._on_row)
        left.addWidget(self.list, 1)
        buttons = QtWidgets.QGridLayout()
        buttons.addWidget(push_button("Thêm API dịch", lambda: self.add("translate"), "add"), 0, 0)
        buttons.addWidget(push_button("Thêm API TTS", lambda: self.add("tts"), "add"), 0, 1)
        self.duplicate_button = push_button("Nhân bản", self.duplicate, "copy")
        self.delete_button = push_button("Xoá", self.delete, "delete")
        buttons.addWidget(self.duplicate_button, 1, 0)
        buttons.addWidget(self.delete_button, 1, 1)
        left.addLayout(buttons)
        layout.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        self.editor_title = QtWidgets.QLabel()
        right.addWidget(self.editor_title)
        self.editor = ProfileEditor(self.kinds)
        self.empty_hint = hint("Thêm API dịch hoặc API TTS để bắt đầu cấu hình.")
        right.addWidget(self.empty_hint)
        right.addWidget(scroll_wrap(self.editor), 1)
        right.addWidget(hint("Số luồng: để trống để dùng Cài đặt; 0 hoặc 1 gửi từng yêu cầu, từ 2 gửi song song."))
        test_row = QtWidgets.QHBoxLayout()
        self.test_button = push_button("Kiểm tra kết nối", self.test, "run", "Primary")
        test_row.addWidget(self.test_button)
        self.test_result = QtWidgets.QLabel()
        self.test_result.setWordWrap(True)
        test_row.addWidget(self.test_result, 1)
        right.addLayout(test_row)
        layout.addLayout(right, 3)
        self.save_timer = QtCore.QTimer(self, singleShot=True, interval=500, timeout=self._save)
        self.editor.changed.connect(self._on_changed)
        self.player = QtMultimedia.QMediaPlayer(self)
        self.audio = QtMultimedia.QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.refresh()

    @property
    def profiles(self) -> list[ProviderProfile]:
        s = self.state.settings
        return s.translate_profiles if self.kind == "translate" else s.tts_profiles

    @property
    def default_id(self) -> str:
        s = self.state.settings
        return s.default_translate_profile if self.kind == "translate" else s.default_tts_profile

    def refresh(self, select: str | None = None) -> None:
        current = select or (self.editor.profile.id if self.editor.profile else None)
        self.list.blockSignals(True)
        self.list.clear()
        self.entries = [(kind, p) for kind in ("translate", "tts") for p in getattr(self.state.settings, f"{kind}_profiles")]
        for kind, profile in self.entries:
            label = self._label(profile, kind)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, profile.id)
            self.list.addItem(item)
        ids = [p.id for _kind, p in self.entries]
        row = ids.index(current) if current in ids else (0 if ids else -1)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self._refresh_active_profiles()
        self._on_row(row)

    def _refresh_active_profiles(self) -> None:
        for category, combo in self.active_profiles.items():
            profiles = getattr(self.state.settings, f"{category}_profiles")
            combo.set_items([(p.id, p.name) for p in profiles], keep=False)
            combo.set_value(getattr(self.state.settings, f"default_{category}_profile"))
            combo.setEnabled(bool(profiles))
            combo.setPlaceholderText("Chưa có API")

    def _switch_api(self, category: str) -> None:
        profile_id = self.active_profiles[category].value()
        profiles = getattr(self.state.settings, f"{category}_profiles")
        if not any(p.id == profile_id for p in profiles):
            return
        setattr(self.state.settings, f"default_{category}_profile", profile_id)
        self.state.save_settings()
        self.refresh(profile_id)

    def _label(self, profile: ProviderProfile, category: str | None = None) -> str:
        category = category or self.kind
        default_id = getattr(self.state.settings, f"default_{category}_profile")
        label = "Dịch" if category == "translate" else "TTS"
        return f"{label} · {profile.name}" + ("  · Đang dùng" if profile.id == default_id else "")

    def _on_row(self, row: int) -> None:
        self.player.stop()
        self.player.setSource(QtCore.QUrl())
        profile = None
        if 0 <= row < len(self.entries):
            self.kind, profile = self.entries[row]
            self.kinds = TRANSLATE_KINDS if self.kind == "translate" else TTS_KINDS
        self.editor.set_profile(profile, self.kinds)
        self.empty_hint.setVisible(profile is None)
        self.editor_title.setText("Cấu hình API dịch" if self.kind == "translate" else "Cấu hình API lồng tiếng (TTS)")
        for button in (self.duplicate_button, self.delete_button, self.test_button):
            button.setEnabled(profile is not None)
        self.test_result.clear()

    def _on_changed(self) -> None:
        item = self.list.currentItem()
        profile = self.editor.profile
        if item and profile:
            item.setText(self._label(profile))
        self._refresh_active_profiles()
        self.save_timer.start()

    def _save(self) -> None:
        self.state.save_settings()

    def add(self, category: str) -> None:
        kind = "openai" if category == "translate" else "openai_speech"
        profile = ProviderProfile(name="API dịch mới" if category == "translate" else "API TTS mới", kind=kind)
        getattr(self.state.settings, f"{category}_profiles").append(profile)
        if not getattr(self.state.settings, f"default_{category}_profile"):
            setattr(self.state.settings, f"default_{category}_profile", profile.id)
        self.state.save_settings()
        self.refresh(profile.id)

    def duplicate(self) -> None:
        profile = self.editor.profile
        if not profile:
            return
        copy = clone(profile)
        copy.id = new_id()
        copy.name += " (bản sao)"
        self.profiles.append(copy)
        self.state.save_settings()
        self.refresh(copy.id)

    def delete(self) -> None:
        profile = self.editor.profile
        if not profile or not confirm(self, "Xoá provider", f"Xoá “{profile.name}”?"):
            return
        self.profiles.remove(profile)
        if self.default_id == profile.id:
            setattr(self.state.settings, f"default_{self.kind}_profile", self.profiles[0].id if self.profiles else "")
        self.state.save_settings()
        self.refresh()

    def test(self) -> None:
        profile = self.editor.profile
        if not profile:
            return
        self._save()
        snapshot = clone(profile)
        category = self.kind
        self.test_button.setEnabled(False)
        self.test_result.setText("Đang kiểm tra…")

        def current() -> bool:
            return self.editor.profile is not None and self.editor.profile.id == snapshot.id

        if category == "translate":
            def work(_p, _s):
                return create_translator(snapshot).test()

            def done(result) -> None:
                if not current():
                    return
                self.test_button.setEnabled(True)
                self.test_result.setText(f"✓ Hoạt động: “你好，世界” → “{result}”")
        else:
            target = Path(tempfile.mkdtemp(prefix="tts_test_", dir=temp_dir())) / "test"

            def work(_p, _s):
                return create_tts(snapshot).test(target)

            def done(result) -> None:
                if not current():
                    return
                self.test_button.setEnabled(True)
                self.test_result.setText(f"✓ Hoạt động — đang phát thử ({Path(result).name})")
                self.player.setSource(QtCore.QUrl.fromLocalFile(str(result)))
                self.player.play()

        def fail(message: str) -> None:
            if not current():
                return
            self.test_button.setEnabled(True)
            self.test_result.setText(f"✕ {message[:400]}")

        run_background(work, done, fail)


class ProvidersPage(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(title_label(
            "Cấu hình API",
            "Chuyển API tại mục API đang sử dụng. Chọn cấu hình trong danh sách bên trái để chỉnh sửa.",
        ))
        self.panel = ApiProfilesPanel(state)
        layout.addWidget(self.panel, 1)
