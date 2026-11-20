from __future__ import annotations

from PyQt6 import QtWidgets

from ...core.models import BatchSettings, SubtitleStyle, clone
from .audio_panel import ORIGINAL_MODES
from .common import ColorButton, ComboBox, DoubleSpinBox, FontCombo, SpinBox, form_layout, hint
from .style_editor import POSITIONS, StylePreview


class BatchSettingsDialog(QtWidgets.QDialog):
    """Các cài đặt chính của Editor cho một lượt chạy hàng loạt."""

    def __init__(self, settings: BatchSettings, presets: list[SubtitleStyle], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt chung cho video")
        self.resize(780, 540)
        self._settings = clone(settings)
        self._presets = clone(presets)
        self._style = clone(settings.style)
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(hint(
            "Áp dụng cho tất cả video khi chạy hàng loạt, hoặc chỉ các video đã chọn. "
            "Cài đặt được lưu theo project; các thay đổi có hiệu lực ở lần chạy tiếp theo."
        ))

        audio_box = QtWidgets.QGroupBox("Âm thanh video gốc")
        audio = QtWidgets.QHBoxLayout(audio_box)
        self.original_mode = ComboBox(ORIGINAL_MODES)
        self.original_mode.set_value(settings.original_mode)
        self.original_volume = SpinBox(0, 200, settings.original_volume, " %")
        self.original_volume.setToolTip(
            "0% tắt tiếng, 100% bằng bản gốc, tối đa 200%. "
            "10% giảm âm gốc 40 dB, 100% bằng bản gốc. Âm lượng TTS được chỉnh riêng."
        )
        audio.addWidget(QtWidgets.QLabel("Chế độ"))
        audio.addWidget(self.original_mode, 1)
        audio.addWidget(QtWidgets.QLabel("Âm lượng"))
        audio.addWidget(self.original_volume)
        layout.addWidget(audio_box)

        self.hide_subtitles = QtWidgets.QCheckBox("Ẩn phụ đề mới đã dịch")
        self.hide_subtitles.setChecked(not settings.subtitles)
        layout.addWidget(self.hide_subtitles)
        layout.addWidget(hint(
            "Bật để không chèn phụ đề mới vào video xuất ra. Vẫn dịch và tạo giọng đọc như bình thường; "
            "không xóa phụ đề có sẵn trong hình ảnh video gốc."
        ))
        columns = QtWidgets.QHBoxLayout()
        self.style_box = QtWidgets.QGroupBox("Kiểu và vị trí phụ đề")
        form = form_layout()
        self.style_box.setLayout(form)
        self.preset = ComboBox([("", "Tùy chỉnh")] + [(str(i), p.name) for i, p in enumerate(presets)])
        self.position = ComboBox(POSITIONS)
        self.margin_v = DoubleSpinBox(0, 45, 7, 0.5, 1, " %")
        self.font = FontCombo()
        self.size = DoubleSpinBox(8, 300, 56, 1, 0, " px")
        self.bold = QtWidgets.QCheckBox("Chữ đậm")
        self.color = ColorButton()
        self.bg_enabled = QtWidgets.QCheckBox("Bật hộp nền sau chữ")
        self.bg_opacity = SpinBox(0, 100, 60, " %")
        self.bg_opacity.setToolTip("0% trong suốt hoàn toàn, 100% đục hoàn toàn.")
        form.addRow("Mẫu phụ đề", self.preset)
        form.addRow("Vị trí", self.position)
        form.addRow("Lề dọc", self.margin_v)
        form.addRow("Phông chữ", self.font)
        form.addRow("Cỡ chữ (ở 1080p)", self.size)
        form.addRow("Màu chữ", self.color)
        form.addRow("", self.bold)
        form.addRow("", self.bg_enabled)
        form.addRow("Độ đục hộp nền", self.bg_opacity)
        columns.addWidget(self.style_box, 1)
        preview_box = QtWidgets.QGroupBox("Xem trước phụ đề")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        self.preview = StylePreview()
        self._preview_text = self.preview.text
        self.preview.setMinimumWidth(280)
        preview_layout.addWidget(self.preview, 1)
        preview_layout.addWidget(hint("Cỡ chữ và lề tự co giãn theo độ phân giải video."))
        columns.addWidget(preview_box, 1)
        layout.addLayout(columns, 1)
        layout.addWidget(hint("Các tùy chọn chi tiết khác vẫn chỉnh trong Editor."))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setText("Lưu và dùng khi chạy")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Hủy")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_style()
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.position.currentIndexChanged.connect(self._edit_style)
        self.margin_v.valueChanged.connect(self._edit_style)
        self.font.currentFontChanged.connect(self._edit_style)
        self.size.valueChanged.connect(self._edit_style)
        self.bold.toggled.connect(self._edit_style)
        self.color.colorChanged.connect(self._edit_style)
        self.bg_enabled.toggled.connect(self._edit_style)
        self.bg_opacity.valueChanged.connect(self._edit_style)
        self.original_mode.currentIndexChanged.connect(self._update_controls)
        self.hide_subtitles.toggled.connect(self._update_controls)
        self._update_controls()

    def _load_style(self) -> None:
        self._loading = True
        style = self._style
        self.position.set_value(style.position)
        self.margin_v.setValue(style.margin_v)
        self.font.set_value(style.font_family)
        self.size.setValue(style.font_size)
        self.bold.setChecked(style.bold)
        self.color.set_value(style.text_color)
        self.bg_enabled.setChecked(style.bg_enabled)
        self.bg_opacity.setValue(style.bg_opacity)
        index = next((str(i) for i, preset in enumerate(self._presets) if preset == style), "")
        self.preset.set_value(index)
        self.preview.set_style(style)
        self._loading = False
        self._update_controls()

    def _apply_preset(self, *_args) -> None:
        if self._loading or not self.preset.value():
            return
        self._style = clone(self._presets[int(self.preset.value())])
        self._load_style()

    def _edit_style(self, *_args) -> None:
        if self._loading:
            return
        style = self._style
        style.position = self.position.value()
        style.margin_v = self.margin_v.value()
        style.font_family = self.font.value()
        style.font_size = self.size.value()
        style.bold = self.bold.isChecked()
        style.text_color = self.color.value()
        style.bg_enabled = self.bg_enabled.isChecked()
        style.bg_opacity = self.bg_opacity.value()
        self.preset.set_value("")
        self.preview.set_style(style)
        self._update_controls()

    def _update_controls(self, *_args) -> None:
        self.original_volume.setEnabled(self.original_mode.value() != "mute")
        show_subtitles = not self.hide_subtitles.isChecked()
        self.style_box.setEnabled(show_subtitles)
        self.preview.text = self._preview_text if show_subtitles else ""
        self.preview.update()
        self.bg_opacity.setEnabled(self.bg_enabled.isChecked())

    def settings(self) -> BatchSettings:
        settings = clone(self._settings)
        settings.enabled = True
        settings.original_mode = self.original_mode.value()
        settings.original_volume = self.original_volume.value()
        settings.subtitles = not self.hide_subtitles.isChecked()
        settings.style = clone(self._style)
        return settings
