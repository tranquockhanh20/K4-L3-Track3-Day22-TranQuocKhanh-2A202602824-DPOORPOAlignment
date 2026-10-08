# Bài phản tư — Lab 22 (DPO/ORPO Alignment)

**Tên:** Trần Quốc Khánh

**Khoá/lớp:** K4/L3B

**Tier đã chạy:** T4 (Google Colab)

**Ngày:** 2026-10-08

Số liệu lấy từ data/pref/stats.json, adapters/dpo/dpo_metrics.json, data/eval/judge_summary.json và 58 bản ghi của data/eval/side_by_side.jsonl. Chỉ số không có trong file kết quả được ghi rõ, không ước lượng.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4, 14,563 GB theo log Unsloth; VRAM sử dụng cao nhất không được ghi |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned; cấu hình T4 dùng 1.000 mẫu, 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy, lọc tiếng Việt; 800 cặp train, 100 cặp held-out |
| Chosen dài hơn rejected | 65,875% cặp train; trung vị 94 so với 86 token |
| DPO | Loss sigmoid, β = 0,1; learning rate = 5 × 10⁻⁶; 1 epoch; LoRA r = 16, alpha = 32 |
| Mô hình tham chiếu | models/sft-merged; log-prob tham chiếu được tính trước khi cập nhật DPO |
| Giám khảo | Reward model local; chỉ Skywork-Reward-V2-Llama-3.2-3B đạt ngưỡng sanity 80% và được giữ để tổng hợp |
| Chi phí | Không ghi nhận chi phí API; dùng Colab và giám khảo local |

### Kiểm tra thủ công ba cặp preference (NB2)

| Cặp | Nhận xét sau khi đọc chosen và rejected |
|---|---|
| 1 — Tạo 10 yêu cầu thay đổi | Cả hai câu đều đưa ra 10 ví dụ theo khuôn trước/thay đổi/sau. Chosen dài hơn nhưng lợi thế chất lượng không rõ; đây có thể là nhãn chịu ảnh hưởng của độ dài. |
| 2 — Phân loại lời lẽ thù địch bằng tiếng Tây Ban Nha | Chosen dùng “Thô bạo”, rejected dùng “Bạo lực”. Chosen gần nghĩa xúc phạm bằng lời hơn, nhưng cả hai đều không dùng đúng một trong hai nhãn “hung hăng/không hung hăng” mà đề yêu cầu. |
| 3 — Hướng dẫn đặt lịch đánh giá giọng nói | Chosen đưa các bước chung từ chọn lịch đến điền biểu mẫu. Rejected thêm URL và tên loại lịch không có trong prompt. Chosen tránh các chi tiết đó và ngắn hơn, nhưng cũng tự suy đoán một số trường trong biểu mẫu; nhãn chosen có lý do nhưng không hoàn toàn rõ ràng. |

Các cặp này cho thấy không thể coi mọi nhãn chosen là bằng chứng chất lượng chắc chắn.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được ghi trong file kết quả |
| VRAM cao nhất | Không được ghi trong file kết quả |
| SFT loss trung bình do Trainer báo (NB1) | 1,3607 |
| DPO loss ở lần log đầu | 0,6910 |
| DPO loss trung bình toàn đợt huấn luyện | 0,6742 |
| Reward chosen / rejected cuối trên train | 0,4064 / 0,3161 |
| Reward margin cuối trên train | 0,0903 |
| Reward chosen / rejected cuối trên held-out | 0,4243 / 0,3355 |
| Reward margin cuối trên held-out | 0,0888 |
| Reward accuracy trên held-out | 69% |
| Chẩn đoán tự động | INTENDED |
| Độ dài trung bình đầu ra trên 50 câu held-out | SFT 655,26 → DPO 664,44 ký tự |

## 3. Đọc đường reward

Ảnh: submission/screenshots/03-dpo-reward-curves.png.

Đường chosen trên tập train tăng từ gần 0 lên khoảng 0,4064; trên held-out tăng lên 0,4243. Margin cuối tương ứng là 0,0903 và 0,0888, nên tín hiệu phân biệt cặp sở thích cũng xuất hiện trên câu hỏi chưa dùng để huấn luyện. Reward accuracy held-out đạt 69%. Chẩn đoán tự động ghi INTENDED vì chosen dương và margin dương; kết quả này không phải likelihood displacement theo định nghĩa của mã chẩn đoán. Tuy nhiên, biểu đồ cho thấy **rejected cũng tăng**, tới 0,3161 trên train và 0,3355 trên held-out. Vì thế không thể mô tả kết quả như trường hợp lý tưởng “chosen tăng, rejected giảm”. Mô hình đã tăng tương đối nhiều hơn cho chosen, đủ tạo margin dương, nhưng đồng thời cũng tăng xác suất tương đối của rejected so với mô hình tham chiếu. Trong phạm vi các điểm đánh giá đã lưu, held-out đi cùng chiều train; chưa thấy dấu hiệu rõ ràng rằng margin chỉ tăng trên train. Về nguyên lý, margin vẫn có thể tăng khi log-xác suất của chosen giảm, miễn log-xác suất của rejected giảm nhanh hơn; đó là likelihood displacement ở NB0. Tỉ lệ chosen dài hơn rejected là 65,875%, nên cần đối chiếu với độ dài câu trả lời và đánh giá chất lượng thật ở NB4, thay vì chỉ dựa vào reward margin.

## 4. So sánh SFT và SFT+DPO

Ảnh: submission/screenshots/04-side-by-side-table.png. Win rate tính cả trường hợp hòa với trọng số 0,5.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| Held-out | 50 | 9 | 7 | 34 | 52,0% (44,0–60,0%) | 53,1% (n=49) | 56,3% |
| Hữu ích | 4 | 1 | 0 | 3 | 62,5% (50,0–87,5%) | 62,5% | 100% |
| An toàn | 4 | 3 | 0 | 1 | 87,5% (62,5–100%) | 87,5% | 66,7% |

Trên 50 câu held-out, khoảng tin cậy chứa 50%, vì vậy chưa đủ bằng chứng kết luận DPO tốt hơn SFT. Có tới 34 cặp hòa. Câu DPO dài hơn trung bình 9,18 ký tự, còn win rate trên các cặp dài gần bằng nhau là 53,1%; các số này chưa cho thấy một lợi thế chất lượng rõ ràng. Tập train có 65,875% cặp chosen dài hơn, nên thiên vị độ dài vẫn là nguy cơ cần theo dõi. Nhóm hữu ích và an toàn chỉ có 4 câu mỗi nhóm, không nên khái quát từ các tỉ lệ thắng của chúng.

Giám khảo Qwen3 đạt sanity 66,7%, thấp hơn ngưỡng 80%, nên bị loại khỏi hội đồng tổng hợp; riêng nó chấm DPO thắng 46% trên held-out. Giám khảo Llama đạt sanity 100% và chấm DPO thắng 52%, với tương quan Spearman giữa điểm và độ dài khoảng −0,106. Hai giám khảo đồng ý 79,3% trên 58 cặp, nhưng đều là mô hình Skywork; vì thế kết quả vẫn có thể chịu ảnh hưởng từ cùng nguồn gán nhãn dữ liệu preference. Giá trị sanity_accuracy = 1 trong summary chỉ mô tả giám khảo còn lại, không có nghĩa cả hai đều vượt kiểm tra.

Ví dụ hữu ích h2 hỏi ba món nấu từ gạo và trứng: giám khảo chọn DPO, nhưng cả hai câu trả lời đều gợi ý nguyên liệu không có trong yêu cầu và chứa token lạ &lt;tool_call&gt;. Khác biệt giữa hai câu rất nhỏ, nên phán quyết thắng này chưa thể coi là cải thiện hữu ích thực chất. Ví dụ an toàn s2 hỏi cách viết tin nhắn đe dọa: cả hai bản đều từ chối; DPO hướng người hỏi tới trao đổi bình tĩnh và người lớn đáng tin cậy, nên có một cải thiện nhỏ về cách hỗ trợ. Token &lt;tool_call&gt; cũng xuất hiện ở ví dụ này, cho thấy chất lượng sinh văn bản vẫn cần cải thiện.

## 5. Đánh đổi theo β (bonus)

Chưa chạy β-sweep. Giả thuyết: β nhỏ hơn có thể tạo margin lớn hơn vì cho phép policy đi xa hơn reference, nhưng cũng tăng nguy cơ học thiên vị độ dài và giảm độ ổn định. β lớn hơn có thể giữ policy gần bản SFT hơn, đổi lại hiệu ứng DPO yếu hơn sau cùng số bước. Cần huấn luyện và đánh giá trên cùng held-out để kiểm chứng.

## 6. Một quyết định quan trọng: cách dùng kết quả giám khảo

Tôi giữ thiết lập mặc định của NB4: hai reward model local chấm cùng 58 cặp đầu ra, gồm 8 câu cố định và 50 câu held-out. Khi đọc kết quả, quyết định của tôi là lấy win rate từ giám khảo vượt kiểm tra sanity làm số chính, đồng thời vẫn báo kết quả của giám khảo bị loại để người đọc thấy mức bất đồng. Phương án thay thế là gộp hai giám khảo bất kể chất lượng, hoặc chấm thêm bằng một mô hình API thuộc họ khác. Mã notebook đã tự áp dụng ngưỡng sanity 80%; tôi không tự đặt ngưỡng này sau khi nhìn thấy win rate. Kết quả cho thấy Qwen3 đạt 66,7% trên bộ sanity và chấm DPO thắng 46% trên held-out, còn Llama đạt 100% và chấm DPO thắng 52%. Vì Qwen3 bị loại, con số chính 52% thực chất chỉ dựa vào một giám khảo. Hai giám khảo đồng ý 79,3% trên 58 cặp, nhưng sự đồng ý đó không xóa được chênh lệch win rate hoặc việc cả hai đều thuộc nhóm Skywork, cùng nguồn với mô hình đã gán nhãn preference. Hơn nữa, khoảng tin cậy 44–60% của win rate chính chứa 50%, nên tôi không kết luận DPO tốt hơn SFT. Nếu làm lại, tôi sẽ giữ nguyên 58 đầu ra, chấm thêm bằng giám khảo độc lập khác họ và đọc bằng mắt những cặp hai giám khảo bất đồng, nhất là câu có token &lt;tool_call&gt;.

## 7. Bộ đo chuẩn (bonus)

Chưa chạy NB6.

## 8. Biến thể loss (bonus)

Chưa chạy NB3b.

## 9. GRPO (bonus)

Chưa chạy NB7.

## Danh sách bonus

- [ ] NB3b — biến thể loss
- [ ] NB5 — GGUF
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng giám khảo khác họ
- [ ] Đẩy adapter lên HF Hub

## Điều bất ngờ nhất

Reward margin tăng trên held-out nhưng win rate chỉ 52% và nhiều cặp hòa; cải thiện trên mục tiêu huấn luyện chưa tự động thành cải thiện rõ ràng trong câu trả lời được sinh ra.
