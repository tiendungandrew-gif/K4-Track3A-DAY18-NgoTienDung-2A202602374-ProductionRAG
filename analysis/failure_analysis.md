# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Ngô Tiến Dũng  
**MSSV:** 2A202602374  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7690 | 0.9248 | +0.1558 |
| Answer Relevancy | 0.7436 | 0.8086 | +0.0650 |
| Context Precision | 0.9118 | 0.8176 | -0.0942 |
| Context Recall | 0.9474 | 0.8337 | -0.1137 |

> **Nhận xét tổng quan:**  
> - **Faithfulness tăng mạnh (+15.58%, đạt 0.9248):** Nhờ cơ chế Contextual Enrichment (M5) và Cross-Encoder Reranking (M3), các đoạn văn được làm sạch và xếp hạng chuẩn xác hơn, triệt tiêu gần như hoàn toàn hiện tượng LLM bịa đặt (hallucination).  
> - **Answer Relevancy tăng (+6.50%, đạt 0.8086):** Mô hình sinh câu trả lời tập trung trực tiếp vào trọng tâm câu hỏi của người dùng hơn so với Naive Baseline.  
> - **Context Precision & Recall:** Sự đánh đổi có kiểm soát khi chuyển từ việc nạp toàn bộ các đoạn thô sang việc chỉ lọc lấy đúng top 3 child chunks có liên kết parent context để tối ưu hóa context window và độ trễ.

---

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.7200, Avg: 0.7300)
- **Error Tree:** Output sai (trả về không tìm thấy) → Context thiếu thông tin đầy đủ do câu hỏi phức hợp đa ý (multi-hop / multi-intent) → Query retriever chỉ lấy được 1 trong 2 tài liệu (hoặc chính sách nghỉ phép, hoặc bảng lương).
- **Root cause:** Câu hỏi yêu cầu tổng hợp thông tin từ 2 nguồn văn bản khác nhau (`chinh_sach_nghi_phep` và `bang_luong_2024`). Với top-3 reranking, các đoạn của tài liệu có điểm tương đồng cao hơn đã chiếm hết slot, đẩy tài liệu thứ hai ra ngoài.
- **Suggested fix:** Áp dụng kỹ thuật **Query Decomposition** (tách câu hỏi thành 2 sub-queries: "ngày phép Senior 9 năm thâm niên" và "khoảng lương Senior") rồi gộp kết quả tìm kiếm, hoặc tăng top-k rerank lên 5 khi phát hiện liên từ "và".

---

### #2
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.7200, Avg: 0.7835)
- **Error Tree:** Output sai → Context không chứa mệnh đề phủ định cho đối tượng thử việc → Tầng tìm kiếm lexical match mạnh từ khóa "bảo hiểm sức khỏe PVI" nhưng chỉ kéo về các điều khoản áp dụng cho nhân viên chính thức.
- **Root cause:** Tài liệu mô tả bảo hiểm PVI tập trung nói về quyền lợi nhân viên chính thức, trong khi quy định loại trừ thử việc nằm ở văn bản thỏa ước lao động / quy chế thử việc riêng.
- **Suggested fix:** Bổ sung metadata phân loại đối tượng (`target_audience: probation | official`) và áp dụng **HyQA** (sinh trước câu hỏi giả định "nhân viên thử việc có được PVI không?") để gắn trực tiếp vào chunk.

---

### #3
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.7200, Avg: 0.7875)
- **Error Tree:** Output sai → Context chưa gom đủ cả 2 điều kiện (thẩm quyền duyệt chi 30 triệu và thủ tục phê duyệt kỹ thuật của phòng CNTT) → LLM thận trọng từ chối trả lời để tránh hallucination.
- **Root cause:** Quy định hạn mức tài chính và quy định phê duyệt kỹ thuật thiết bị CNTT nằm ở 2 mục riêng biệt trong sổ tay mua sắm.
- **Suggested fix:** Sử dụng kỹ thuật **Parent-Document Retrieval** với kích thước parent lớn hơn (tăng từ 2048 lên 3072) để khi một child chunk match, toàn bộ quy trình mua sắm đi kèm được nạp vào context của LLM.

---

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Đoạn văn nằm trong tài liệu bang_luong_2024.md... Nhân viên trong thời gian thử việc được nhận **85% lương** của cấp bậc tương ứng.
- **Worst metric:** Answer Relevancy (0.7200, Avg: 0.8042)
- **Error Tree:** Output chưa hoàn chỉnh (thiếu phép tính ra con số 17.000.000 VNĐ) → Context đã có đủ mức lương Junior (10-20tr) và tỷ lệ 85% → Prompt generation chưa kích hoạt khả năng suy luận số học (arithmetic reasoning).
- **Root cause:** System prompt hiện tại chỉ yêu cầu "Trả lời CHỈ dựa trên context", khiến mô hình có xu hướng trích xuất nguyên văn tỷ lệ phần trăm thay vì thực hiện phép nhân $85\% \times 20.000.000$.
- **Suggested fix:** Cải tiến prompt template với kỹ thuật Chain-of-Thought ngắn: "Nếu câu hỏi yêu cầu giá trị cụ thể và ngữ cảnh cung cấp tỷ lệ phần trăm kèm hạn mức, hãy thực hiện phép tính toán rõ ràng trước khi kết luận".

---

### #5
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Trong tài liệu mat_khau_v1.md, mật khẩu phải được thay đổi mỗi 90 ngày, còn trong tài liệu mat_khau_v2.md, mật khẩu phải được thay đổi mỗi 120 ngày.
- **Worst metric:** Answer Relevancy (0.7653, Avg: 0.8134)
- **Error Tree:** Output phân vân giữa 2 phiên bản tài liệu xung đột → Context bị nhiễu do nạp cả phiên bản cũ v1.0 và phiên bản mới v2.0 → Retriever không nhận biết được tính thời gian/hiệu lực của văn bản.
- **Root cause:** Xung đột phiên bản tài liệu (Version conflict). Cả 2 file `mat_khau_v1.md` và `mat_khau_v2.md` đều có trong database nhưng hệ thống chưa có cơ chế lọc văn bản hết hiệu lực.
- **Suggested fix:** Bổ sung metadata `version` và `is_active: bool` khi ingest văn bản; trước khi retrieval, tự động lọc `is_active == True` hoặc prompt LLM ưu tiên tài liệu có version cao nhất khi có xung đột quy định.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
`Bao lâu phải đổi mật khẩu một lần?`

**Error Tree walkthrough:**
1. **Output đúng?** → Không hoàn toàn. Mô hình nêu cả 2 mốc 90 ngày và 120 ngày thay vì khẳng định mốc 120 ngày theo chính sách hiện hành.
2. **Context đúng?** → Bị nhiễu tài liệu cũ. Context trả về gồm cả chunk từ `mat_khau_v1.md` (chính sách cũ đã hết hạn) và `mat_khau_v2.md` (chính sách mới).
3. **Query rewrite OK?** → Câu hỏi rất ngắn gọn, rõ ràng ("Bao lâu phải đổi mật khẩu một lần?"). Vấn đề không nằm ở khâu hiểu query mà nằm ở khâu quản trị tri thức (Knowledge Base Governance).
4. **Fix ở bước:** Tầng **Metadata Ingestion & Pre-retrieval Filtering**:
   - Khi đánh chỉ mục, gán metadata `document_status: "deprecated"` cho `mat_khau_v1.md` và `"active"` cho `mat_khau_v2.md`.
   - Áp dụng bộ lọc `Filter(status == "active")` trong Qdrant và BM25 trước khi tìm kiếm.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Triển khai cơ chế **Temporal / Version-aware Filtering**: Tự động so sánh số hiệu phiên bản và ngày ban hành để tự động loại bỏ các văn bản cũ bị thay thế khỏi candidate pool.
- Bổ sung **Chain-of-Thought Prompting** cho khâu Generation để giải quyết dứt điểm các câu hỏi đòi hỏi tính toán số học (như tính lương thử việc 85% và tính phạt quá hạn tạm ứng 2%/tháng).
