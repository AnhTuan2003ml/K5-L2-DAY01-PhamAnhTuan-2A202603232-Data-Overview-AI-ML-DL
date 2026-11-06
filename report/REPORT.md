# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026  
**Runtime Colab:** GPU Tesla T4  
**Python / PyTorch / Ultralytics:** Python `3.13.16`; PyTorch `2.11.0+cu130`; Ultralytics `8.4.145`  
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`  
**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`  
**Thay đổi so với notebook nguồn:** Không.

Bài thực hành sử dụng ba checkpoint YOLO11 để phân tích kết quả phân loại ảnh, phát hiện vật thể và phân đoạn đối tượng. Dữ liệu được ghi nhận trong các file JSON và hình ảnh trực quan thuộc thư mục `day1_lab_outputs/`.

### Từ cần nhớ

| Từ | Nghĩa |
| --- | --- |
| Prediction | Kết quả dự đoán của mô hình, không phải nhãn chuẩn đã xác nhận. |
| Ground truth | Nhãn chuẩn được gán theo guideline và kiểm tra chất lượng. |
| Taxonomy | Hệ thống danh mục lớp dùng để xác định ý nghĩa và mã lớp. |
| Checkpoint | Tệp trọng số mô hình đã được huấn luyện, định dạng `.pt`. |
| Guideline | Bộ quy tắc thống nhất cách gán nhãn dữ liệu. |
| Annotator | Người trực tiếp gán nhãn hoặc chỉnh sửa nhãn. |
| Reviewer / QC | Người kiểm tra chất lượng, tính chính xác và nhất quán của nhãn. |
| Escalation | Chuyển trường hợp chưa rõ quy tắc đến Lab Coach để được thống nhất cách xử lý. |
| Model score | Mức tin cậy của dự đoán do mô hình đưa ra. |

## 1. Phân loại ảnh

**Dữ liệu:** `day1_lab_outputs/classification_predictions.json`, `day1_lab_outputs/visuals/classification_top5.png`.  
**Ảnh phân tích:** `sample_id = traffic`, kết quả xếp hạng 1.

**1. Record hạng 1**

- `class_id`: `468`
- `class_name`: `cab`
- `rank`: `1`
- `score`: `0.510915`
- `taxonomy_name`: `ImageNet-1K`

**2. Phạm vi của kết quả phân loại**

Record trên là dự đoán cho toàn bộ ảnh `traffic`, không phải cho một vật thể riêng lẻ. Mô hình phân loại ảnh trả về danh sách các lớp được xếp hạng theo score; kết quả không chứa bounding box hay tọa độ để xác định vị trí từng vật thể.

**3. Nguồn gốc danh sách lớp**

Danh sách lớp thuộc taxonomy `ImageNet-1K` của checkpoint `yolo11n-cls.pt`. Mô hình lựa chọn trong các lớp đã được quy định trong hệ thống này, không tự tạo tên lớp mới trong quá trình dự đoán.

**4. Ý nghĩa của `class_id`, `class_name` và `taxonomy_name`**

`class_id` là mã định danh lớp; `class_name` là tên lớp có thể đọc và diễn giải; `taxonomy_name` xác định hệ thống danh mục mà lớp đó thuộc về. Việc lưu cả ba trường giúp xác định nhãn chính xác và tránh nhầm lẫn khi các bộ dữ liệu sử dụng cách đánh số hoặc tên lớp khác nhau.

**5. Quy tắc chọn một nhãn cho ảnh có nhiều đối tượng**

Guideline quy định việc chọn nhãn dựa trên chủ thể chính hoặc nội dung trọng tâm của ảnh, đồng thời xác định thứ tự ưu tiên khi xuất hiện nhiều đối tượng. Trường hợp ảnh `traffic` có nhiều loại phương tiện, annotator chọn một lớp theo tiêu chí thống nhất của bộ dữ liệu, thay vì mặc định sử dụng nhãn `cab` do mô hình dự đoán.

**6. Phân biệt model score và ground truth**

Score `0.510915` biểu thị mức tin cậy của mô hình đối với lớp `cab`. Ground truth là nhãn được người gán theo guideline và được kiểm tra chất lượng. Một dự đoán có score cao vẫn có khả năng sai, vì vậy model score không thay thế được ground truth.

## 2. Phát hiện vật thể

**Dữ liệu:** `day1_lab_outputs/detection_predictions.json`, `day1_lab_outputs/visuals/detection_predictions.png`.  
**Ảnh phân tích:** `sample_id = kitchen`.

**1. Record được chọn**

- `class_name`: `person`
- `score`: `0.912625`
- `bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`
- `bbox_width`: `113.58` pixel
- `bbox_height`: `279.68` pixel

**2. Diễn giải bounding box**

Bounding box xác định người đứng bên phải ảnh `kitchen`. Góc trên bên trái của hộp ở tọa độ `(385.33, 69.24)`, góc dưới bên phải ở `(498.92, 348.92)`. Theo dữ liệu JSON, hộp có chiều rộng `113.58` pixel và chiều cao `279.68` pixel. Hệ tọa độ sử dụng gốc tại góc trên bên trái của ảnh, trục `x` tăng về bên phải và trục `y` tăng xuống dưới.

**3. Số lượng prediction theo hai ngưỡng**

| Ngưỡng confidence | Số prediction trên ảnh `kitchen` |
| --- | ---: |
| `0.35` | 11 |
| `0.60` | 6 |

Trong các prediction của ảnh `kitchen`, có 11 kết quả đạt score từ `0.35` trở lên và 6 kết quả đạt score từ `0.60` trở lên. Việc tăng ngưỡng loại bỏ 5 kết quả có score thấp hơn `0.60`.

**4. Công việc của reviewer khi giảm ngưỡng**

Ngưỡng thấp làm tăng số bounding box được giữ lại, bao gồm cả các kết quả có độ tin cậy thấp. Reviewer phải kiểm tra thêm khả năng gán sai lớp, hộp trùng lặp, hộp không khớp vật thể và những đối tượng bị bỏ sót, từ đó xác định các nhãn phải chỉnh sửa hoặc loại bỏ.

**5. Quy tắc vẽ bounding box**

Mỗi vật thể thuộc phạm vi gán nhãn được bao bằng một hình chữ nhật ôm sát phần vật thể nhìn thấy, hạn chế vùng nền dư thừa và không vượt ra ngoài ranh giới ảnh. Các vật thể tách biệt được gán hộp riêng theo cùng một quy tắc.

**6. Trường hợp vật thể bị cắt mép ảnh và hướng xử lý**

Ở mép trái ảnh `kitchen`, một người chỉ xuất hiện một phần cơ thể, được mô hình nhận diện là `person` với score `0.610931` và hộp `[0.12, 263.18, 61.55, 310.78]`. Đây là trường hợp vật thể bị cắt bởi ranh giới ảnh. Trường hợp này được xử lý bằng guideline: gán nhãn khi phần hiện diện đủ để nhận diện và chỉ vẽ box quanh phần nằm trong ảnh, không suy đoán phần cơ thể ngoài khung hình. Nếu không đủ căn cứ xác định lớp, trường hợp được chuyển đến Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Dữ liệu:** `day1_lab_outputs/segmentation_predictions.json`, `day1_lab_outputs/visuals/segmentation_prediction.png`.  
**Ảnh phân tích:** `sample_id = kitchen`.

**1. Record được chọn**

- `instance_id`: `kitchen-001`
- `class_name`: `person`
- `score`: `0.899318`
- `polygon_point_count`: `348`
- Năm điểm đầu của `polygon_xy`: `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]`

**2. Thông tin bổ sung của polygon so với bounding box**

Bounding box mô tả vị trí và kích thước bao ngoài bằng hình chữ nhật. Polygon sử dụng nhiều điểm để xác định đường biên vật thể, thể hiện rõ hơn hình dạng thực tế, các phần lồi lõm và khu vực vật thể chiếm trong ảnh.

**3. Ý nghĩa của `instance_id`**

`instance_id = kitchen-001` là mã phân biệt một đối tượng cụ thể trong kết quả phân đoạn của ảnh `kitchen`. Nhiều đối tượng có thể cùng lớp `person` nhưng có `instance_id` khác nhau. Mã này không phải `class_id` và cũng không phải mã theo dõi một đối tượng xuyên suốt các khung hình hoặc video.

**4. Quy tắc vẽ biên mask**

Mask được vẽ sát ranh giới phần vật thể thực sự nhìn thấy, không lấn sang nền hoặc vật thể bên cạnh. Mỗi đối tượng riêng biệt có mask riêng; phần bị che khuất không được tự suy đoán và tô bổ sung khi guideline không quy định.

**5. Trường hợp ranh giới bị che khuất và hướng xử lý**

Trong ảnh `kitchen`, instance `kitchen-009` được dự đoán là `person`, score `0.431438`, nằm sát mép trái ảnh, phía sau khu vực mặt bàn và các vật dụng. Phần cơ thể xuất hiện trong ảnh bị giới hạn nên ranh giới với vật thể phía trước khó phân định. Tôi áp dụng guideline vẽ mask theo phần nhìn thấy; những trường hợp không xác định được ranh giới hoặc mức độ che khuất hợp lệ được chuyển đến Lab Coach để thống nhất cách gán nhãn.

## 4. Vòng đời và kiểm tra chất lượng

Quy trình dữ liệu trong hệ thống học máy gồm các giai đoạn:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Bài thực hành sử dụng prediction từ mô hình YOLO11 đã huấn luyện sẵn để nhận diện sự khác biệt giữa kết quả dự đoán và nhãn chuẩn trong quy trình sản xuất dữ liệu.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp đại diện cho toàn bộ ảnh `traffic`, thuộc taxonomy được quy định. | Ảnh `traffic` có nhiều phương tiện; kết quả Top-1 là `cab` với score `0.510915`, trong khi việc chọn nhãn đại diện cho toàn cảnh có thể không thống nhất. | Xác định chủ thể chính, gán một nhãn đúng taxonomy và ghi nhận trường hợp chưa rõ quy tắc. | Kiểm tra nhãn đã chọn có phù hợp nội dung ảnh và nguyên tắc phân loại toàn ảnh hay không. |
| Phát hiện vật thể | Mỗi vật thể có nhãn lớp và một bounding box xác định vị trí, kích thước. | Một người xuất hiện một phần ở mép trái ảnh `kitchen`; mô hình dự đoán `person`, score `0.610931`, với tọa độ trái `x = 0.12` pixel. | Gán lớp nếu đủ căn cứ, vẽ hộp ôm phần cơ thể xuất hiện trong ảnh, không mở rộng ra ngoài khung hình. | Kiểm tra nhãn lớp, độ khít của bounding box, hộp trùng lặp và vật thể bị bỏ sót. |
| Instance segmentation | Mỗi đối tượng có một nhãn lớp và một polygon/mask riêng thể hiện vùng ảnh của đối tượng. | Ở ảnh `kitchen`, instance `kitchen-007` được dự đoán là `oven`, score `0.489667`; vùng mask tại khu vực bếp nấu bên trái có ranh giới và phạm vi lớp khó xác định. | Xác định lớp theo taxonomy, phân tách mask theo phần vật thể nhìn thấy và chuyển ca chưa rõ quy tắc đến Lab Coach. | Kiểm tra tính đúng lớp, độ chính xác đường biên, vùng mask bị tràn hoặc thiếu và yêu cầu chỉnh sửa nhãn không đạt. |

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu**

Chỉ sử dụng dữ liệu được phép trong phạm vi bài thực hành, bảo đảm thông tin nguồn và không công khai dữ liệu cá nhân, dữ liệu nội bộ hoặc dữ liệu nhạy cảm trên Colab hay GitHub.

**2. Xử lý dữ liệu ngoài phạm vi**

Tôi dừng xử lý, không tải dữ liệu lên môi trường công khai và báo ngay cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Danh mục tệp của bộ báo cáo dành cho thư mục `report/`:

- [x] `REPORT.md`
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp link repository trên LMS
