<p align="center"><img src="assets/logo.svg" width="128" alt="ReviewTrans Studio"></p>

# ReviewTrans Studio

Công cụ dịch, lồng tiếng và chèn phụ đề tự động cho video review phim, với giao diện kiểu phần mềm dựng video.

## Tính năng

- **Project nhiều video**: mỗi project (một bộ phim) chứa nhiều video xếp theo thứ tự tập, dùng chung
  **ngữ cảnh**: tóm tắt cốt truyện, nhân vật + cách xưng hô, thuật ngữ. Sau mỗi video, LLM tự bổ sung ngữ cảnh.
  Mục nào đã “khoá” sẽ không bị ghi đè.
- **Editor**: bảng câu thoại (sửa trực tiếp, tách/gộp/khoá câu), khung phát libmpv (tự chuyển sang Qt Multimedia
  nếu chưa có libmpv), inspector, timeline nhiều track (layer, phụ đề, lồng tiếng, âm gốc, nhạc nền, video).
- **API dịch và lồng tiếng**: quản lý chung trên một trang. Dịch qua API kiểu OpenAI
  (OpenAI, DeepSeek, OpenRouter, LM Studio…), API kiểu Gemini, Anthropic Claude. Hỗ trợ nhiều key xoay vòng.
- **Lồng tiếng**: vBee, API kiểu OpenAI `/v1/audio/speech`, Custom API. Có thể đặt giọng riêng
  cho từng nhân vật hoặc từng câu. Sửa câu nào chỉ tạo lại lồng tiếng câu đó.
  Số luồng mặc định là `0` (gửi tuần tự); ô số luồng trong từng API để trống sẽ dùng giá trị trong Cài đặt.
- **Bật/tắt** phụ đề, lồng tiếng, âm gốc, nhạc nền, từng layer.
- **Âm thanh gốc**: chỉnh cố định 0–200% cho toàn bộ video, hoặc tự giảm theo TTS với thời gian giảm/tăng dần;
  mặc định giữ ở 20% cả trong khoảng nghỉ giữa các câu. Đổi âm lượng chỉ cần trộn lại, không tạo lại TTS.
- **Style phụ đề**: font, cỡ, màu chữ, viền, bóng, hộp nền, vị trí, lề, số ký tự mỗi dòng. Chỉnh, lưu và áp preset ngay trong Editor.
- **Layer**: ảnh (logo), chữ, vùng che (làm mờ / pixel hoá / tô màu) để che chữ cứng của video gốc.
  Kéo thả trên khung phát và trên timeline.
- **Hàng đợi**: chạy hàng loạt Nhận dạng → Dịch → Cập nhật ngữ cảnh → Lồng tiếng → Trộn âm → Xuất video.
- **ASR tiếng Trung**: Paraformer-zh + FSMN-VAD + CT-Punc + CAM++, chia SRT theo timestamp từ, khoảng nghỉ và bằng chứng đổi giọng.
  App chỉ dùng Paraformer cho tiếng Trung/Anh; ngôn ngữ khác có thể nhập SRT.
  Xem [cài đặt và giới hạn](docs/asr-paraformer.md).
- **Tăng tốc phần cứng**: xuất video bằng GPU (NVIDIA NVENC, AMD AMF, Intel Quick Sync, Media Foundation), tự dò và
  chạy thử trước khi dùng, lỗi thì tự chuyển về CPU. Nhận dạng giọng nói bằng GPU qua PyTorch CUDA (NVIDIA).
- Trang riêng cho **Providers**, **Tài nguyên** (tải ffmpeg, bộ Paraformer, libmpv),
  **Công cụ** (ghép video), **Cài đặt**.

⬇️ **[Tải bản mới nhất (installer / portable)](https://github.com/dominhhieu1405/ReviewTrans/releases/latest)**

📖 **[Hướng dẫn sử dụng đầy đủ (có ảnh minh hoạ)](docs/huong-dan-su-dung.md)**

## Chạy từ mã nguồn

Sau khi cài thư viện, nhấp đúp `run.vbs` để mở ứng dụng trực tiếp, không mở PowerShell hay cửa sổ dòng lệnh.
File này dùng Python trong `.venv` và vẫn chạy đúng khi tạo shortcut ra Desktop.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

Lần đầu mở, vào trang **Tài nguyên** để tải FFmpeg, bộ Paraformer (cho tiếng Trung)
và libmpv (player).
Bản chạy từ mã nguồn dùng thư mục `data/` ngay trong thư mục dự án, kể cả khi mở từ terminal ở vị trí khác:

```
data/settings.json        cấu hình, provider và danh sách project
data/bin/                 FFmpeg, FFprobe, libmpv
data/models/              bộ FunASR
data/funasr-runtime/      môi trường PyTorch/FunASR riêng, cài trong Tài nguyên
data/fonts/               font bổ sung
data/tmp/                 file tải và giải nén tạm, âm thanh thử
data/projects/            nơi tạo project mặc định (gồm cache và lồng tiếng)
```

Đây là dữ liệu mới, không tự nhập cấu hình, model hay project từ `~/.video_translation_studio`.
Có thể đổi vị trí bằng biến môi trường `REVIEWTRANS_DATA_DIR` trước khi mở app.
Bản portable dùng `data/` cạnh file exe; bản cài đặt vẫn dùng thư mục dữ liệu người dùng.

## Cấu trúc

```
main.py                   điểm khởi chạy
reviewtrans/core/         mô hình dữ liệu, provider, pipeline (không phụ thuộc widget)
reviewtrans/ui/           giao diện PyQt6 (player, timeline, các trang)
tests/                    pytest (có test render thật bằng ffmpeg)
legacy/                   mã nguồn bản cũ, chỉ để tham khảo
```

Dữ liệu một project nằm trong thư mục của nó (`project.json`, `context.json`, `videos/<id>/…`, `output/`),
có thể sao chép hoặc sao lưu nguyên thư mục.

## Test

```bash
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```
