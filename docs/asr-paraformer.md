# Paraformer-zh thay Whisper cho video tiếng Trung

ReviewTrans chỉ dùng **Paraformer-zh + FSMN-VAD + CT-Punc** để nhận dạng tiếng Trung/Anh.
Project/video cũ chọn Whisper được chuyển sang Paraformer khi mở. Ngôn ngữ khác cần
nhập SRT có sẵn. Các câu đã lưu không thay đổi nếu chưa chạy lại bước Nhận dạng.

## Cơ sở lựa chọn

- [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) là model offline cho
  tiếng Trung và tiếng Anh, dùng âm thanh 16 kHz. Đây là ứng viên phù hợp cho video
  review tiếng Trung; chưa có kết quả đo trên video của người dùng để khẳng định
  chính xác hơn Whisper trong mọi trường hợp. Tốc độ do nhà phát triển công bố
  không phải tốc độ đã đo của ReviewTrans.
- [FSMN-VAD](https://huggingface.co/funasr/fsmn-vad) phát hiện vùng có lời nói và chia
  âm thanh dài trước ASR. VAD không tự nhận dạng chữ hoặc căn timestamp từng chữ.
- [CT-Punc](https://huggingface.co/funasr/ct-punc) bổ sung dấu câu; kết hợp timestamp
  nhận dạng để FunASR trả về `sentence_info` khi bật `sentence_timestamp=True`.
- [Hợp đồng Python SDK](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md)
  và [AutoModel](https://github.com/modelscope/FunASR/blob/main/funasr/auto/auto_model.py)
  mô tả timestamp theo mili giây. App chuyển sang giây một lần, giữ timeline gốc,
  không áp ánh xạ VAD của whisper.cpp và không ước lượng thời gian bằng độ dài chữ.

Paraformer không phải bộ nhận dạng tiếng Việt/đa ngôn ngữ thay thế toàn bộ Whisper.
`zh-tw` vẫn là nhận dạng tiếng Trung; model không đảm bảo trả văn bản phồn thể.
CAM++ được dùng để so sánh giọng hai phía ranh giới từ trước khi chia SRT;
không gán tên nhân vật hoặc tự chọn giọng TTS.

## Ranh giới lời thoại

CT-Punc dự đoán dấu câu từ văn bản toàn tập; dấu phẩy không đảm bảo hết lượt nói,
và một câu có thể chứa lời của nhiều người. App căn lại theo timestamp từng từ:

- Tách tại khoảng nghỉ từ 800 ms, kể cả khi CT-Punc không đặt dấu câu.
- Với khoảng nghỉ từ 100 ms, so sánh hai cửa sổ âm thanh tối đa 1,5 giây bằng
  [CAM++](https://huggingface.co/funasr/campplus). Tách nếu cosine similarity ≤ 0,32.
- Nối qua dấu phẩy khi khoảng cách ≤ 180 ms và similarity ≥ 0,45; giới hạn mỗi
  câu ghép ở 7 giây / 80 ký tự. Giữ dấu câu và toàn bộ chữ đã nhận dạng.
- Không dùng cửa sổ dưới 400 ms hoặc dưới hai token; không suy đoán khi thiếu
  timestamp hoặc không khớp chữ. Ngưỡng giọng là heuristic, không phải xác suất.

Các ngưỡng được kiểm tra trên cache/âm thanh S04E001 và S04E010. Lời chồng lấn,
đổi người không nghỉ, tiếng quá ngắn hoặc nhạc nền vẫn có thể cần sửa trong Editor.
Không sử dụng nội dung bản dịch để đoán lại thời gian nói.

Model CAM++ thêm khoảng 28 MB, được tải cùng bộ nhận dạng trong Tài nguyên.
Bản đã cài ba model cần tải bổ sung trước khi chạy ASR mới. Nhận dạng không tự
tải model. Chạy lại **Nhận dạng → Dịch → TTS** cho video cần cập nhật; SRT đã xuất
và dữ liệu đã biên tập không tự đổi theo mã nguồn mới.

## Cài đặt và sử dụng

1. Vào **Tài nguyên**, tải **FFmpeg + FFprobe** nếu chưa có.
2. Trong khung **Paraformer-zh**, bấm **Cài bộ nhận dạng + tải model**.
   Chọn **Tự động CPU / CUDA**, **NVIDIA CUDA** hoặc **CPU**. Trên Windows bộ cài
   tải Python 3.11.9 độc lập, không cần Python có sẵn. Cần Internet lần đầu và
   tối thiểu 8 GiB trống cho CPU hoặc 16 GiB cho CUDA (ngưỡng dự phòng bao gồm
   file tải và giải nén, không phải dung lượng tải chính xác). Bộ cài kiểm tra
   dung lượng trước khi tải; không giữ cache pip và xóa bộ tải Python sau khi giải nén.
3. Chọn ngôn ngữ gốc **Tiếng Trung**, chọn bộ nhận dạng
   **Paraformer-zh + FSMN-VAD + CT-Punc**.
4. Chạy bước **Nhận dạng**. Câu xuất hiện trong Editor để sửa, dịch và lồng tiếng
   theo quy trình hiện có. SRT gốc lưu tại `videos/<id>/cache/asr.srt` trong project.
   SRT xuất từ Editor vẫn sử dụng các câu đang được biên tập.

Bộ cài tạo `data/funasr-runtime/` riêng, không cài PyTorch vào môi trường giao diện.
Model nằm ở `data/models/funasr/{paraformer-zh,fsmn-vad,ct-punc,campplus}/`. Bản portable/bản
cài đặt dùng thư mục dữ liệu theo quy tắc hiện có của app. Script worker và danh
sách phụ thuộc được đóng gói cùng exe. Bản ZIP có ASR còn kèm sẵn `asr-runtime/`
(Python + PyTorch + FunASR) và `asr-models/` ngay cạnh `ReviewTrans.exe`.

Để cài trong app mà không nhân đôi model khi đóng gói: dùng ZIP portable thông thường
(không bật `--bundle-asr`), giải nén vào thư mục có quyền ghi trên ổ D, giữ
`portable.txt` cạnh `ReviewTrans.exe`, rồi bấm cài trong trang Tài nguyên. Runtime,
bốn model và file tạm đều được lưu dưới `data/` trong thư mục đó. Có thể di chuyển
cả thư mục sau khi đóng app. Đây là một thư mục chung, không phải một file exe duy nhất.
Nếu chạy mã nguồn, dữ liệu mặc định cũng ở `data/` cạnh mã nguồn. Biến môi trường
`REVIEWTRANS_DATA_DIR` nếu được đặt sẽ ưu tiên nơi lưu khác; trang Tài nguyên hiển thị
đường dẫn thực tế để kiểm tra trước khi cài. Bản cài đặt không có `portable.txt`
vẫn dùng thư mục dữ liệu người dùng theo quy tắc của app.

Bộ cài **Tự động** ưu tiên PyTorch/torchaudio `2.8.0+cu126` khi phát hiện NVIDIA
với driver hỗ trợ CUDA 12.6 trở lên; còn lại dùng `2.8.0+cpu`. FunASR là `1.4.26`.
Trang Tài nguyên hiển thị bản PyTorch và GPU thực tế đã kiểm tra. Cache pip,
file tạm và cache ASR mới được đặt trong thư mục dữ liệu app, cùng ổ đĩa với bản portable.
Nhận dạng sau khi tải đủ model không cần tải model lại. Không bật thực thi mã
Python từ model (`trust_remote_code=False`), không tự cài/cập nhật khi đang nhận dạng.

Nếu nhận dạng lỗi hoặc thiếu timestamp, app báo lỗi và giữ trạng thái bước chưa
hoàn tất. App báo lỗi trực tiếp khi Paraformer không chạy được.
Bấm **Dừng** trong hàng đợi sẽ kết thúc tiến trình nhận dạng riêng.

## GPU NVIDIA hoặc môi trường có sẵn

Paraformer dùng PyTorch CPU hoặc CUDA.
Tùy chọn **Tự động** dùng CUDA khi `torch.cuda.is_available()` trả về True;
**CPU** luôn dùng CPU. Chọn **CUDA** mà PyTorch không có CUDA sẽ báo trong log
và chạy CPU. Cấu hình Vulkan cũ được chuyển sang Tự động khi nạp cài đặt.

Để dùng NVIDIA, chọn **NVIDIA CUDA** trong khung Paraformer ở trang Tài nguyên
rồi cài lại bộ nhận dạng. Driver cần tương thích CUDA 12.6. Các wheel có kèm
thư viện CUDA cần thiết; không yêu cầu cài riêng CUDA Toolkit.
Xem [các bản wheel chính thức](https://pytorch.org/get-started/previous-versions/#v280).
Máy dùng bản ZIP kèm ASR CUDA có thể chạy CPU khi không có NVIDIA tương thích.
Nút cài sẽ tạo bản cập nhật trong `data/funasr-runtime/`; bản này được ưu tiên
trước môi trường đi kèm ZIP.

Có thể chỉ định Python riêng trước khi mở app:

```powershell
$env:REVIEWTRANS_FUNASR_PYTHON = 'D:\asr-env\Scripts\python.exe'
```

Môi trường này cần PyTorch/torchaudio tương thích và `requirements-asr.txt`.
Biến này chỉ đổi Python dùng để nhận dạng; nút cài vẫn quản lý môi trường mặc định.

Kiểm tra môi trường và model mà không mở giao diện:

```powershell
data\funasr-runtime\python.exe scripts\funasr_worker.py --check
data\funasr-runtime\python.exe scripts\funasr_worker.py --download --models data\models\funasr
```

Khi so sánh với Whisper, dùng cùng WAV và video có bản chép đúng; đo CER cho tiếng
Trung, lỗi tên riêng, câu bị thiếu/thêm, độ lệch timestamp đầu/cuối câu, và thời gian
chạy trên cùng thiết bị. Một mẫu công khai chỉ kiểm tra việc tích hợp, không đủ để
kết luận chất lượng trên phim có nhạc nền, lời chồng lấn hoặc giọng địa phương.

## Kết quả kiểm chứng ngày 14/11/2026

- Đã cài và nhập thực tế FunASR `1.4.26`, PyTorch/torchaudio `2.8.0+cpu` trên
  Windows/Python 3.11; tải đủ ba model từ Hugging Face.
- Sau đó đã nâng cấp sang Python portable 3.11.9 và PyTorch/torchaudio
  `2.8.0+cu126`, chạy tensor CUDA và cả pipeline nhận dạng thật trên **Quadro T1000
  4 GB**, driver 581.32. Kết quả xác nhận `device=cuda:0`; model và timestamp vẫn
  hoạt động. Mẫu ngắn mất 21,734 giây gồm nạp thư viện/model; CUDA không đảm bảo
  nhanh hơn CPU trên mẫu ngắn vì còn thời gian khởi tạo.
- Dùng [mẫu tiếng Trung công khai của FunASR](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav),
  lặp hai lần, thêm 2 giây im lặng đầu, 7 giây giữa và 2 giây cuối: tổng 22,093 giây.
  Luồng app chạy thật trên CPU mất 18,514 giây, gồm nạp model và tách WAV.
  SRT có hai câu tại `2,920–7,510` và `15,479–20,070` giây; khoảng nghỉ giữa câu
  được giữ trên timeline gốc. Hai lần nhận dạng mẫu lặp vẫn khác nhau về chữ,
  vì vậy đây là xác nhận hoạt động/timestamp, chưa phải bằng chứng giảm CER so với Whisper.
- 68 kiểm thử ASR, timestamp, cấu hình và phần cứng đạt. Chạy rộng hơn: tổng
  200 kiểm thử đạt (gồm 8 kiểm thử đường dẫn chạy riêng để cô lập FFmpeg).
  Hai kiểm thử còn lỗi ngoài phần tích hợp: `test_stop_during_tts_saves_partial_results_and_resumes_force_mode`
  kỳ vọng chuỗi TTS chưa có dấu chấm, và `test_main_window_smoke` kỳ vọng profile
  dịch thứ hai trong khi cấu hình mặc định hiện chỉ tạo một profile.
