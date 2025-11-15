# 🏙️ Chatbot Gợi Ý Du Lịch TP.HCM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*Hệ thống chatbot thông minh kết hợp AI và Recommendation System để gợi ý địa điểm du lịch tại Thành phố Hồ Chí Minh*

[Tính năng](#-tính-năng) • [Cài đặt](#-cài-đặt) • [Sử dụng](#-sử-dụng) • [Demo](#-demo)

</div>

---

## 📖 Giới thiệu

Chatbot Gợi Ý Du Lịch TP.HCM là một ứng dụng web thông minh sử dụng kết hợp **Recommendation System** và **Large Language Model (LLM)** để cung cấp trải nghiệm tư vấn du lịch cá nhân hóa. Hệ thống có khả năng:

- 🎯 Đề xuất địa điểm du lịch phù hợp với nhu cầu người dùng
- 🔍 Tìm kiếm các địa điểm tương tự dựa trên đặc điểm
- 💬 Trò chuyện tự nhiên như một hướng dẫn viên du lịch thực sự
- 📝 Lưu trữ và quản lý lịch sử hội thoại

## ✨ Tính năng

### 🎯 Gợi ý địa điểm thông minh
- Đề xuất các địa điểm du lịch nổi bật khi người dùng hỏi chung chung
- Phân tích ngữ cảnh câu hỏi để đưa ra gợi ý phù hợp

### 🔍 Hệ thống khuyến nghị tương đồng
- Sử dụng thuật toán **TF-IDF** và **Cosine Similarity**
- Tìm kiếm và đề xuất địa điểm có đặc điểm tương tự
- Hỗ trợ tìm kiếm theo tên địa điểm cụ thể

### 💬 Trò chuyện tự nhiên
- Tích hợp mô hình LLM thông qua **Ollama**
- Xử lý các câu hỏi không liên quan đến du lịch
- Trả lời một cách tự nhiên và thân thiện

### 📊 Quản lý lịch sử
- Lưu trữ toàn bộ cuộc hội thoại trong `chat_history.json`
- Chức năng xóa lịch sử qua giao diện
- Theo dõi và phân tích hành vi người dùng

## 🛠️ Công nghệ sử dụng

### Backend
- **Flask** - Web framework
- **Python 3.8+** - Ngôn ngữ lập trình chính

### Machine Learning & AI
- **Scikit-learn** - TF-IDF, Cosine Similarity
- **Pandas** - Xử lý và phân tích dữ liệu
- **NumPy** - Tính toán số học
- **Ollama** - Triển khai LLM (Llama 3, Gemma, v.v.)

### Frontend
- **HTML5/CSS3** - Giao diện người dùng
- **JavaScript (ES6+)** - Xử lý tương tác
- **Fetch API** - Giao tiếp với Backend

## 📁 Cấu trúc dự án

```
Chatbot-LLM/
│
├── .vscode/                    # Cấu hình VS Code
│
├── templates/
│   └── index.html             # Giao diện web chính
│
├── Chatbot.py                 # Flask server & business logic
├── chat_history.json          # Lưu trữ lịch sử hội thoại
├── Dataset.csv                # Dữ liệu địa điểm du lịch
├── tfidf_matrix.pkl           # Ma trận TF-IDF đã huấn luyện
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt           # Dependencies
└── README.md                  # Tài liệu hướng dẫn
```

### 📄 Mô tả các file chính

| File | Mô tả |
|------|-------|
| `Chatbot.py` | Server Flask chính, xử lý API `/chat` và `/clear_history` |
| `index.html` | Giao diện người dùng, nơi tương tác với chatbot |
| `Dataset.csv` | Dữ liệu thô về địa điểm và đánh giá |
| `tfidf_vectorizer.pkl` | Model vectorizer đã được train |
| `tfidf_matrix.pkl` | Ma trận đặc trưng TF-IDF |
| `chat_history.json` | Lưu trữ lịch sử chat (tự động tạo) |

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **Ollama**: Bắt buộc để chạy LLM
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Storage**: ~2GB cho models

### Bước 1: Cài đặt Ollama

1. Truy cập [ollama.com](https://ollama.com) để tải về
2. Cài đặt theo hướng dẫn cho hệ điều hành của bạn
3. Tải model LLM (khuyến nghị `gemma:2b` cho hiệu suất tốt):

```bash
ollama pull gemma:2b
```

**Các model khác:**
```bash
# Model nhẹ, phù hợp máy yếu
ollama pull gemma:2b

# Model mạnh hơn, cần RAM cao
ollama pull llama3
ollama pull llama3:8b
```

4. Kiểm tra Ollama đang chạy:
```bash
ollama list
```

### Bước 2: Clone dự án

```bash
# Sử dụng Git
git clone https://github.com/your-username/Chatbot-LLM.git
cd Chatbot-LLM

# Hoặc tải ZIP và giải nén
```

### Bước 3: Tạo môi trường ảo

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Bước 4: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Nội dung `requirements.txt`:**
```
flask==2.3.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
```

### Bước 5: Kiểm tra dữ liệu

Đảm bảo các file sau tồn tại:
- ✅ `Dataset.csv`
- ✅ `tfidf_vectorizer.pkl`
- ✅ `tfidf_matrix.pkl`

## ▶️ Chạy ứng dụng

### Khởi động server

```bash
python Chatbot.py
```

Kết quả mong đợi:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Truy cập ứng dụng

Mở trình duyệt và truy cập:
```
http://127.0.0.1:5000
```

hoặc

```
http://localhost:5000
```

## 💡 Sử dụng

### Các loại câu hỏi được hỗ trợ

#### 1️⃣ Gợi ý chung
```
❓ "Gợi ý cho tôi vài địa điểm du lịch"
❓ "Tôi nên đi đâu chơi ở TP.HCM?"
❓ "Giới thiệu địa điểm thú vị"
```

#### 2️⃣ Tìm địa điểm tương tự (Kích hoạt Recommendation System)
```
🎯 "Gợi ý địa điểm Thảo Cầm Viên"
🎯 "Cho tôi biết về Dinh Độc Lập"
🎯 "Tôi muốn đến địa đạo Củ Chi"
🎯 "Địa điểm như Bảo tàng Chứng tích Chiến tranh"
```

#### 3️⃣ Trò chuyện thông thường (Kích hoạt LLM)
```
💬 "Xin chào"
💬 "Bạn là ai?"
💬 "Thời tiết hôm nay thế nào?"
💬 "1 + 1 bằng mấy?"
```

### Xóa lịch sử

- Nhấn nút **"Xóa Lịch Sử"** trên giao diện
- Toàn bộ cuộc hội thoại trong `chat_history.json` sẽ bị xóa

## 🎨 Demo

### Giao diện chính
```
┌─────────────────────────────────────┐
│  Chatbot Gợi Ý Du Lịch TP.HCM      │
├─────────────────────────────────────┤
│                                     │
│  🤖: Xin chào! Tôi có thể giúp gì? │
│  👤: Gợi ý địa điểm Thảo Cầm Viên  │
│  🤖: Dựa trên Thảo Cầm Viên, tôi   │
│      gợi ý: Bảo tàng Lịch sử...    │
│                                     │
├─────────────────────────────────────┤
│  [Nhập tin nhắn...]  [Gửi] [Xóa LS]│
└─────────────────────────────────────┘
```

## 🔧 Tùy chỉnh

### Thay đổi model LLM

Trong file `Chatbot.py`, tìm và sửa:

```python
# Thay 'gemma:2b' bằng model khác
model = "llama3:8b"
```

### Thêm địa điểm mới

Chỉnh sửa file `Dataset.csv`:
```csv
Địa điểm,Mô tả,Rating
Landmark 81,Tòa nhà cao nhất Việt Nam,4.5
...
```

Sau đó chạy lại script để tạo lại TF-IDF matrix.

## ⚠️ Xử lý lỗi thường gặp

### Lỗi: "Ollama not found"
**Giải pháp:**
```bash
# Kiểm tra Ollama đã cài đặt
ollama --version

# Khởi động Ollama service
ollama serve
```

### Lỗi: "File not found: tfidf_matrix.pkl"
**Giải pháp:**
- Đảm bảo file tồn tại trong thư mục gốc
- Chạy lại script training (nếu có)

### Lỗi: "Port 5000 already in use"
**Giải pháp:**
```bash
# Thay đổi port trong Chatbot.py
app.run(port=5001)
```

## 📊 Hiệu suất

| Model | RAM Usage | Response Time |
|-------|-----------|---------------|
| gemma:2b | ~2GB | ~1-2s |
| llama3 | ~4GB | ~2-4s |
| llama3:8b | ~8GB | ~3-6s |


## 📝 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 👥 Tác giả

**Your Name**
- GitHub: https://github.com/itsminhsang
- Email: minhsanglaitran2309@gmail.com

## 🙏 Lời cảm ơn

- [Ollama](https://ollama.com) - LLM infrastructure
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning tools

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho một star nhé! ⭐**

Made with ❤️ in Ho Chi Minh City

</div>
