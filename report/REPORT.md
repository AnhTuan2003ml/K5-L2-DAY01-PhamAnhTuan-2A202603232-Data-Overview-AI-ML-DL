# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** Chưa xác nhận từ dữ liệu xuất (điền ngày thực tế trên Colab).

**Runtime Colab:** Chưa xác nhận CPU/GPU từ các file JSON/PNG (kiểm tra tại Colab).

**Python / PyTorch / Ultralytics:** Python: chưa xác nhận; PyTorch: chưa xác nhận; Ultralytics: `8.4.145` (được ghi trong JSON).

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (đối chiếu trường `model_file` trong ba JSON).

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (đối chiếu trường `score_threshold` trong JSON).

**Thay đổi so với notebook nguồn:** Chưa xác minh được từ ZIP; cần đối chiếu notebook đã chạy. Không tự khẳng định “Không” khi chưa kiểm tra.

> **Cách dùng mẫu này:** Nội dung dưới đây giữ nguyên **6 mục và thứ tự câu hỏi của `REPORT_TEMPLATE.md`**. Ba JSON và ba ảnh PNG là kết quả prediction, **không phải ground truth** đã được người kiểm duyệt xác nhận. Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân vào báo cáo/output.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** [`day1_lab_outputs/classification_predictions.json`](day1_lab_outputs/classification_predictions.json) và [`visuals/classification_top5.png`](day1_lab_outputs/visuals/classification_top5.png).  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1: `class_id`; `class_name`; `rank`; `score`; `taxonomy_name`.**  
   `class_id` = `468`; `class_name` = `cab`; `rank` = `1`; `score` = `0.510915`; `taxonomy_name` = `ImageNet-1K`.

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao?**  
   Đây là prediction **cấp toàn ảnh**: model Classification xếp hạng một lớp cho ảnh `traffic`, không xuất bbox hay tọa độ một vật thể. Ảnh có nhiều phương tiện nhưng kết quả Top-1 chỉ là lớp được model ưu tiên cho toàn cảnh.

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**  
   Danh sách lớp thuộc taxonomy **ImageNet-1K** gắn với checkpoint `yolo11n-cls.pt`; model dự đoán giữa các lớp đã định nghĩa chứ không tự đặt tên lớp mới khi suy luận.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**  
   `class_id` định danh lớp theo chỉ số; `class_name` giúp người đọc hiểu; `taxonomy_name` cho biết hệ danh mục đang dùng. Chỉ giữ tên lớp dễ gây nhầm khi các taxonomy khác nhau sử dụng tên giống hoặc khác cách đánh số.

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**  
   Guideline phải quy định tiêu chí chọn **chủ thể chính** (ví dụ: trọng tâm ảnh, đối tượng nổi bật hoặc mục đích nhãn của bộ dữ liệu), ưu tiên khi có nhiều lớp hợp lệ và quy trình chuyển ca khi không thể quyết định nhất quán. Với ảnh `traffic` có nhiều loại xe, không tự coi `cab` là ground truth chỉ vì model xếp đầu.

6. **Vì sao không được lấy `score` của model làm ground truth?**  
   `score = 0.510915` chỉ phản ánh mức độ ưu tiên của model đối với lớp `cab` trong lần dự đoán này. Ground truth là nhãn được người gán theo guideline rồi kiểm tra QC; score cao vẫn có thể dự đoán sai.

## 2. Phát hiện vật thể

**Mở file:** [`day1_lab_outputs/detection_predictions.json`](day1_lab_outputs/detection_predictions.json) và [`visuals/detection_predictions.png`](day1_lab_outputs/visuals/detection_predictions.png).  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record: `class_name`; `score`; `bbox_xyxy`; `bbox_width`; `bbox_height`.**  
   Record chọn: `class_name` = `person`; `score` = `0.912625`; `bbox_xyxy` = `[385.33, 69.24, 498.92, 348.92]`; `bbox_width` = `113.58` px; `bbox_height` = `279.68` px.

2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**  
   Box bao người đứng phía bên phải ảnh bếp: góc trên-trái tại `(385.33, 69.24)` px; góc dưới-phải tại `(498.92, 348.92)` px (gốc tọa độ ảnh ở góc trên-trái, trục x sang phải, trục y xuống dưới). Theo JSON, hộp rộng **113.58 px**, cao **279.68 px**. Khi lấy hiệu từ các tọa độ đã làm tròn độc lập, chiều rộng ra 113.59 px, lệch 0.01 px do làm tròn.

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**  
   Chọn **0.35** và **0.60** để có thể đối chiếu ngay từ file JSON: tại `0.35` có **11 prediction** cho `kitchen`; trong 11 record này có **6 prediction có `score ≥ 0.60`**. Đây là **kết quả đếm/lọc từ JSON được tạo ở 0.35**, không phải số liệu đã đọc trực tiếp từ ô chạy lại notebook ở ngưỡng 0.60. Cần kiểm tra ô so sánh ngưỡng trên Colab để xác nhận con số hiển thị của notebook; ZIP không chứa output riêng cho ngưỡng 0.20.

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**  
   Reviewer phải kiểm tra thêm hộp score thấp, loại hộp sai/nhầm lớp/trùng lặp, đối chiếu vật thể bị bỏ sót và yêu cầu sửa bbox không ôm sát vật thể. Khối lượng QC tăng vì số prediction có thể tăng.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh).**  
   **Quy tắc đề xuất:** Với mỗi vật thể hợp lệ theo taxonomy, vẽ **một hộp chữ nhật nhỏ nhất ôm sát phần vật thể nhìn thấy**, không bao thừa nền và không kéo hộp ra ngoài mép ảnh; cùng một cách xử lý phải được áp dụng nhất quán cho mọi ảnh.

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Dùng guideline hay escalation? Vì sao?**  
   Bên **trái, sát mép ảnh** có một người chỉ lộ phần thân/cánh tay, được Detection dự đoán `person` với score `0.610931`, box `[0.12, 263.18, 61.55, 310.78]`. Đây là trường hợp **cắt mép ảnh**. Tôi chọn ghi quy tắc vào **guideline**: chỉ gán nhãn khi phần nhìn thấy đủ nhận diện và vẽ box theo phần nằm trong ảnh, không suy đoán phần bị cắt. Nếu mức độ nhìn thấy không đủ rõ theo quy định hiện có, chuyển **escalation** cho Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** [`day1_lab_outputs/segmentation_predictions.json`](day1_lab_outputs/segmentation_predictions.json) và [`visuals/segmentation_prediction.png`](day1_lab_outputs/visuals/segmentation_prediction.png).  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record: `instance_id`; `class_name`; `score`; `polygon_point_count`; 3–5 điểm đầu của `polygon_xy`.**  
   `instance_id` = `kitchen-001`; `class_name` = `person`; `score` = `0.899318`; `polygon_point_count` = `348`; 5 điểm đầu `polygon_xy` = `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]`.

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**  
   Box chỉ cho biết vùng hình chữ nhật bao quanh; polygon cho biết đường viền nhiều điểm của phần vật thể mà model tách ra, mô tả dáng người và các chỗ lồi/lõm chi tiết hơn.

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — hiểu thế nào?**  
   `kitchen-001` là mã để phân biệt **một instance prediction** với những instance khác trong output ảnh `kitchen`. `class_id = 0` mới là mã lớp `person`; `instance_id` không thể hiện danh tính một người và không chứng minh đó vẫn là cùng người trong ảnh/video khác.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.**  
   **Quy tắc đề xuất:** Mask đi sát **đường biên phần vật thể thực sự nhìn thấy**, không tô lan lên nền hay vật thể kế bên; mỗi vật thể tách biệt có mask riêng. Với phần bị che, không tự vẽ phần khuất trừ khi guideline yêu cầu rõ.

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Dùng guideline hay escalation? Vì sao?**  
   Mask `instance_id = kitchen-009` (`person`, score `0.431438`) nằm sát **mép trái ảnh và vùng mặt bàn/bát**, nên phần người lộ ra rất ít và ranh giới giữa người với các vật ở phía trước khó kiểm tra. Tôi áp dụng guideline về **chỉ vẽ phần nhìn thấy**, nhưng đưa **escalation** nếu chưa có tiêu chuẩn rõ về mức độ che khuất tối thiểu để được tạo một instance; tránh tự suy đoán phần người bị che.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **Một nhãn lớp cho toàn ảnh** `traffic` theo taxonomy/guideline. | Trong `classification_predictions.json`, Top-1 `cab` (score `0.510915`), nhưng ảnh `traffic` có nhiều loại xe; khó chọn một nhãn đại diện, **chưa kết luận model sai**. | Xem toàn ảnh, xác định chủ thể chính theo quy tắc và chọn đúng một lớp; hỏi Lab Coach khi guideline chưa rõ. | Kiểm tra nhãn toàn ảnh có đúng taxonomy và tiêu chí chọn chủ thể chính; không coi Top-1 là ground truth. |
| Phát hiện vật thể | **Mỗi vật thể có một class và một bbox `xyxy`** trong ảnh. | Ảnh `kitchen` có người bị **cắt ở mép trái**: prediction `person`, score `0.610931`, box bắt đầu tại `x = 0.12` px (`detection_predictions.json`). | Gán nhãn nếu phần lộ ra đủ nhận diện; box chỉ ôm phần nằm trong ảnh và ghi ca khó. | Soát đúng lớp, box không vượt mép, không bỏ sót/trùng hộp; yêu cầu rework nếu sai guideline. |
| Instance segmentation | **Mỗi instance có class riêng và mask/polygon riêng** mô tả phần hình của vật thể. | `segmentation_predictions.json` có `kitchen-007` = `oven`, score `0.489667`; mask nằm tại **vùng bếp nấu bên trái** và gặp biên vật thể/đồ đạc, khó xác nhận phần nào thuộc `oven` theo guideline. Đây là **điểm mơ hồ**, chưa phải lỗi đã xác nhận. | Kiểm tra lớp với taxonomy; vẽ mask theo biên nhìn thấy, không tô lan sang đồ vật cạnh bên; escalation khi khó phân định. | Soát lớp và biên mask trên PNG, kiểm tra tràn nền, thiếu vùng, chồng mask; trả rework hoặc xin Lab Coach thống nhất quy tắc. |

**Ghi chú:** Bảng mô tả **ground truth cần được người tạo nhãn xác nhận trong production**, không khẳng định các prediction trong JSON đã là nhãn chuẩn. Không có tệp ground truth thẩm định trong ZIP.

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**  
   Chỉ sử dụng ảnh và dữ liệu đúng phạm vi bài lab, giữ nguyên thông tin nguồn/quyền sử dụng và không tải dữ liệu cá nhân, nội bộ hoặc nhạy cảm lên Colab hay GitHub công khai.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**  
   Tôi dừng xử lý/không upload, giữ nguyên bằng chứng cần thiết trong phạm vi được phép và báo **Lab Coach/giảng viên phụ trách** để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền và được đóng gói; **cần xác nhận sau khi upload GitHub**)
- [x] `day1_lab_outputs/classification_predictions.json` (đã có trong ZIP)
- [x] `day1_lab_outputs/detection_predictions.json` (đã có trong ZIP)
- [x] `day1_lab_outputs/segmentation_predictions.json` (đã có trong ZIP)
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md` (đã có trong ZIP)
- [x] `day1_lab_outputs/visuals/classification_top5.png` (đã có trong ZIP)
- [x] `day1_lab_outputs/visuals/detection_predictions.png` (đã có trong ZIP)
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png` (đã có trong ZIP)
- [ ] Ô validation cuối notebook báo `PASS` (chưa có ảnh/log xác nhận)
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output (đã rà soát file báo cáo và các tệp output đã cung cấp)
- [ ] Đã nộp **link repository** trên LMS

> **Cần làm trước khi nộp:** Xác nhận ngày chạy, CPU/GPU, phiên bản Python/PyTorch và thay đổi notebook (nếu có); kiểm tra ô validation `PASS`; giải nén ZIP, upload nội dung vào thư mục `report/` trên GitHub và nộp link repository trên LMS. Các ô có dấu `[x]` về file cho biết **bộ ZIP đã chứa file**, không có nghĩa là chúng đã được xác nhận trên GitHub.
