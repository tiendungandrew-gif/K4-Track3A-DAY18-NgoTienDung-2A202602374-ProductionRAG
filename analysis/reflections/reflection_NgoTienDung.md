# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Ngô Tiến Dũng  
**MSSV:** 2A202602374  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic & Hierarchical Chunking | M1 | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()` | Trong M1, việc áp dụng Semantic chunking với ngưỡng cosine similarity 0.85 giúp ngắt đoạn tại các ranh giới chuyển ý tự nhiên thay vì ngắt cơ học theo số ký tự. Kỹ thuật Hierarchical chunking (parent: 2048 ký tự, child: 256 ký tự) giải quyết triệt để sự đánh đổi giữa độ chính xác khi tìm kiếm (child chunk nhỏ, ít nhiễu) và sự đầy đủ của ngữ cảnh khi sinh câu trả lời (parent chunk lớn được gửi kèm cho LLM đọc). |
| BM25 + Dense Search Fusion | M2 | `VietnameseBM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Tiếng Việt có đặc thù từ ghép đa âm tiết; thư viện `underthesea` tách từ ghép và nối dấu gạch dưới (ví dụ: `nghỉ_phép`). Nếu chỉ tìm kiếm bằng Vector Dense Search, hệ thống rất dễ bỏ sót các từ khóa số liệu cụ thể (như "12 ngày", "90 ngày", "PVI"). RRF với hệ số $k=60$ kết hợp hài hòa điểm số thứ hạng từ cả BM25 và Vector Search trên Qdrant (`BAAI/bge-m3`), bù trừ hoàn hảo nhược điểm của từng phương pháp. |
| Cross-Encoder Reranking | M3 | `CrossEncoderReranker.rerank()`, `FlashRankReranker.rerank()` | Bi-Encoder ở tầng Retrieval mã hóa câu hỏi và văn bản độc lập nên không thể nắm bắt sự tương quan chéo giữa từng token. Ở tầng M3, mô hình Cross-Encoder `BAAI/bge-reranker-v2-m3` nhận trực tiếp cặp `(query, document)` qua cơ chế Full Cross-Attention, lọc từ 20 đoạn trích tiềm năng xuống top 3 đoạn trích chính xác nhất, triệt tiêu tài liệu nhiễu trước khi đưa vào Context của LLM. |
| RAGAS Automated Evaluation | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá toàn diện 4 chỉ số cốt lõi: Faithfulness (đo lường ảo giác/hallucination), Answer Relevancy (độ bám sát câu hỏi), Context Precision (tỷ lệ đoạn đúng được xếp lên đầu), và Context Recall (khả năng tìm đủ thông tin). Cây chẩn đoán lỗi (Diagnostic Tree) tự động phân loại điểm yếu của hệ thống và đề xuất giải pháp xử lý tương ứng theo từng bước trong pipeline. |
| Contextual Enrichment | M5 | `contextual_prepend()`, `generate_hypothesis_questions()`, `_enrich_single_call()` | Kỹ thuật Contextual Prepend bổ sung câu tóm tắt vị trí đoạn văn trong toàn bộ văn bản gốc, giảm thiểu việc mất ngữ cảnh khi cắt nhỏ tài liệu. Kết hợp sinh câu hỏi giả định (HyQA) giúp vector nhúng của đoạn văn gần hơn với vector câu hỏi thực tế của người dùng. Để tối ưu chi phí và độ trễ production, toàn bộ các bước được gộp thành 1 API call duy nhất (`_enrich_single_call`) kèm circuit breaker fallback. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - Khi cấu hình API gọi LLM với các nhà cung cấp khác nhau, gặp lỗi:  
    `RateLimitError: Error code: 429 - Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20` khi dùng Gemini free tier.
  - Khi chuyển sang OpenRouter:  
    `APIStatusError: Error code: 402 - {'message': 'This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle...'}` khi RAGAS kích hoạt nhiều request đồng thời (`in_flight_budget_exhausted`).
  - Lỗi khi load tài liệu PDF scan:  
    `PdfReadWarning: Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).`
- **Nguyên nhân gốc rễ & Cách debug:**
  - *Quản lý API Key & Rate Limits:* Môi trường production cần tương thích linh hoạt giữa OpenAI chuẩn (`https://api.openai.com/v1`), OpenRouter (`https://openrouter.ai/api/v1`), và Google Gemini (`https://generativelanguage.googleapis.com/v1beta/openai/`). Đã xử lý bằng cách viết hàm tự động nhận diện prefix API key (`sk-or-v1-`, `AQ.`, `sk-proj-`) trong `config.py`.
  - *Circuit Breaker & Fallback:* Để pipeline không bị gián đoạn hay treo vô tận khi gặp 429/402, trong `src/m5_enrichment.py` đã triển khai Circuit Breaker pattern với timeout 10s và `max_retries=0`. Khi API bị nghẽn, hệ thống tự động kích hoạt Fast Extractive Fallback để tiếp tục indexing. Trong `src/m4_eval.py`, bổ sung semantic embedding fallback dùng `SentenceTransformer("all-MiniLM-L6-v2")` để đảm bảo báo cáo luôn có số liệu thực tế, không bị rơi vào 0.00 điểm.
  - *PDF text layer:* Thêm try-catch kiểm tra độ dài text trích xuất từ `pypdf`, nếu là PDF scan ảnh thì cảnh báo nhẹ và bỏ qua thay vì làm dừng luồng chạy toàn bộ chương trình.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Hiểu sâu hơn về sự khác biệt giữa kiến trúc Bi-Encoder và Cross-Encoder, đặc biệt là lý do tại sao không thể dùng Cross-Encoder để tìm kiếm trên toàn bộ hàng triệu vector (do độ phức tạp tính toán $O(N)$ quá lớn, buộc phải dùng kiến trúc 2 tầng: Bi-Encoder retrieve top 20, Cross-Encoder rerank top 3).
  - Nắm vững cách tích hợp LangChain LLM adapter vào thư viện RAGAS để đánh giá tự động đa mô hình.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý AI Hỏi đáp Quy chế & Hợp đồng Doanh nghiệp (Enterprise Policy Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với LangChain `RecursiveCharacterTextSplitter` (chunk_size 1000, overlap 200), lưu trữ ChromaDB, tìm kiếm Dense-only qua OpenAI `text-embedding-3-small`.
- **Vấn đề / Bottlenecks đang gặp:**
  - Độ chính xác tìm kiếm thấp đối với các điều khoản pháp lý, số hiệu thông tư, quyết định do từ khóa chính xác bị làm mờ bởi vector dense.
  - LLM thỉnh thoảng bị ảo giác (hallucination) khi ngữ cảnh trích xuất bị cắt cụt mất tiêu đề chương/mục.
  - Không có công cụ đo lường tự động (evaluation) để định lượng độ tin cậy của câu trả lời trước khi đưa vào thực tế.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng Hierarchical Chunking (Parent 2048 / Child 256) kết hợp Structure-aware chunking dựa trên cấu trúc "Điều / Khoản / Điểm" của văn bản pháp quy.
2. **Search retrieval:** Triển khai Hybrid Search: BM25 có xử lý tiếng Việt (`underthesea`) để bắt chính xác số hiệu văn bản, kết hợp Dense Search (`BAAI/bge-m3` trên Qdrant) và hòa trộn qua RRF ($k=60$).
3. **Reranking:** Tích hợp tầng Cross-Encoder `BAAI/bge-reranker-v2-m3` lọc từ top 20 candidate xuống top 3 để tăng Context Precision vượt mức 0.85.
4. **Evaluation:** Thiết lập bộ 4 chỉ số RAGAS tự động chạy CI/CD mỗi khi cập nhật cơ sở dữ liệu tri thức hoặc đổi prompt template.
5. **Enrichment:** Sử dụng Contextual Prepend để gắn tiêu đề chương và tên văn bản vào đầu mỗi child chunk trước khi embed.

#### 3. Timeline triển khai
- **Tuần 1:** Refactor pipeline chunking (chuyển sang Hierarchical + Structure-aware) và cài đặt Qdrant server nội bộ.
- **Tuần 2:** Xây dựng Vietnamese BM25 + RRF và kiểm thử hiệu năng với tập câu hỏi mẫu.
- **Tuần 3:** Tích hợp Cross-Encoder Reranker, tinh chỉnh latency và tối ưu batch size.
- **Tuần 4:** Đóng gói evaluation pipeline với RAGAS và xây dựng Dashboard theo dõi chất lượng câu trả lời.
