from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import subprocess, json, os, unicodedata, difflib
import re, functools, time

# Flask+API key
app = Flask(__name__)
app.secret_key = "super_secret_key" 

# Từ dừng tiếng Việt
VIETNAMESE_STOP_WORDS = set([
    'là', 'của', 'tôi', 'mình', 'cho', 'và', 'hay', 'những', 'cái', 'này', 'kia', 'ở', 'đó', 'tại', 'với', 'cũng', 'một',
    'nào', 'về', 'thì', 'làm', 'muốn', 'đi', 'xin', 'hỏi', 'có', 'được', 'rồi', 'như', 'bị', 'các', 'sẽ', 'nếu',
    'giá', 'phí', 'vé', 'xe', 'tàu', 'bus', 'mấy', 'giờ', 'thời', 'gian', 'khi', 'nào', 'mùa', 'gì', 'ăn', 'món'
])

# Hàm chuẩn hóa đầu vào
def normalize_text(text):
    nfkd = unicodedata.normalize('NFD', str(text).lower())
    text_no_diacritics = ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')
    text_clean = re.sub(r'[^a-z0-9\s]', '', text_no_diacritics).strip()
    return text_clean

# Hàm bỏ từ dừng
def tokenize(text):
    normalized_input = normalize_text(text)
    tokens = [t for t in normalized_input.split() if t and t not in VIETNAMESE_STOP_WORDS]
    return tokens

# Hàm xác định loại câu
def extract_question_type(text):
    question_types = {
        'location': ['o dau', 'cho nao', 'dia chi', 'duong nao'],
        'transport': ['di bang gi', 'phuong tien'],
        'time': ['mo cua', 'dong cua'],
        'price': ['bao nhieu tien'],
        'activity': ['lam gi', 'hoat dong', 'trai nghiem', 'choi gi'],
        'food': ['dac san', 'nha hang']
    }
    text_norm = normalize_text(text)
    for qtype, keywords in question_types.items():
        if any(kw in text_norm for kw in keywords):
            return qtype
    return 'general'

# Hàm tìm địa điểm trong câu hỏi
def find_place_in_input(user_input_normalized, place_norm_map, cutoff=0.65, min_jaccard=0.3):
    input_norm = user_input_normalized 
    input_tokens = set(input_norm.split())
    if not input_norm:
        return None, None
    best_match_place = None
    best_score = -1
    for norm, original in place_norm_map.items():
        place_tokens = set(norm.split())
        if norm in input_norm:
            print(f"[DEBUG] find_place_in_input: Match 'exact_string' ({original})")
            return original, 'exact_string'
        if place_tokens and input_tokens:
            intersection = len(input_tokens & place_tokens)
            union = len(input_tokens | place_tokens)
            jaccard_sim = intersection / union
            if jaccard_sim > best_score:
                best_score = jaccard_sim
                best_match_place = original
    if best_match_place and best_score >= min_jaccard:
        print(f"[DEBUG] find_place_in_input: Match 'jaccard_token' ({best_match_place}, Score: {best_score})")
        return best_match_place, 'jaccard_token'
    matches = difflib.get_close_matches(input_norm, place_norm_map.keys(), n=1, cutoff=cutoff)
    if matches:
        print(f"[DEBUG] find_place_in_input: Match 'fuzzy_diff' ({place_norm_map[matches[0]]})")
        return place_norm_map[matches[0]], 'fuzzy_diff'
    print("[DEBUG] find_place_in_input: No match found.")
    return None, None

# Dữ liệu đầu vào và Ma trận Tương đồng
DATA_FILE = "Dataset.csv"
try:
    df = pd.read_csv(DATA_FILE, index_col=0, encoding="utf-8-sig")
    df.columns = df.columns.astype(str)
    
# Kiểm tra và xử lý tên địa điểm trùng lặp
    if any(df.columns.duplicated()):
        print("Cảnh báo: Phát hiện tên địa điểm trùng lặp trong dữ liệu. Đang tiến hành đổi tên...")
        cols = pd.Series(df.columns)
        
# Đổi tên các cột trùng lặp bằng cách đánh số
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index
            cols.loc[dup_indices] = [f"{dup} (copy_{i+1})" for i in range(len(dup_indices))]
        
        df.columns = cols.astype(str)
        print(f"Đã xử lý xong. Tổng số địa điểm duy nhất: {len(df.columns)}")
        
# Tính toán Ma trận Tương đồng
    ratings = df.replace(0, np.nan)
    normalized_ratings = ratings.fillna(df.replace(0, np.nan).mean())

    similarity_cosine = pd.DataFrame(
        cosine_similarity(normalized_ratings.T),
        index=ratings.columns,
        columns=ratings.columns
    )

    similarity_pearson = normalized_ratings.T.corr(method='pearson')
    similarity_pearson = similarity_pearson.fillna(0)
    
    similarity_matrix = (0.7 * similarity_cosine + 0.3 * similarity_pearson)
    
    place_norm_map = {normalize_text(col): col for col in df.columns}

except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    df = pd.DataFrame()
    similarity_matrix = pd.DataFrame()
    place_norm_map = {}
#File json thực hiện lưu lịch sử chat
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                content = f.read()
                return json.loads(content) if content else []
        except json.JSONDecodeError:
            print(f"[ERROR] Lỗi đọc JSON từ {HISTORY_FILE}. Trả về lịch sử rỗng.")
            return [] 
    return []

def save_history(data):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            history_to_save = data[-100:]
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi khi lưu lịch sử chat: {e}")

# Tải lịch sử chat ban đầu
chat_history = load_history()
def get_rating_summary(place):
    if str(place) not in df.columns:
        print(f"[DEBUG] get_rating_summary: Lỗi - '{place}' (type: {type(place)}) không tìm thấy trong df.columns.")
        return None
        
    ratings = df[str(place)].replace(0, np.nan)
    avg_rating = ratings.mean()
    rating_count = ratings.count()
    
    return {
        'average': round(avg_rating, 1) if pd.notna(avg_rating) else 0.0,
        'count': rating_count,
        'popularity': 'cao' if (pd.notna(avg_rating) and avg_rating >= 4) else 'trung bình' if (pd.notna(avg_rating) and avg_rating >= 3) else 'thấp'
    }

def recommend(place, top_n=3):
    if str(place) not in similarity_matrix.columns or similarity_matrix.empty:
        print(f"[DEBUG] recommend: Lỗi - '{place}' không có trong similarity_matrix.")
        return []
        
    sims = similarity_matrix[str(place)].sort_values(ascending=False).drop(str(place), errors='ignore')
    
    if sims.empty:
        print("[DEBUG] recommend: Lỗi - Không tìm thấy địa điểm tương đồng (sims is empty).")
        return []

# BƯỚC 1: Chỉ lấy các chỉ mục (tên địa điểm) có tồn tại trong df.columns
    recommendation_indices = sims.index.intersection(df.columns)
    
    if recommendation_indices.empty:
        print("[DEBUG] recommend: Lỗi - Không tìm thấy địa điểm gợi ý nào có trong df.")
        return []
        
    # BƯỚC 2: Tạo df mục tiêu bằng cách sử dụng các tên cột đã được kiểm tra (recommendation_indices)
    df_target = df[recommendation_indices].replace(0, np.nan)
    
    # BƯỚC 3: Tính mean/count.
    all_avg_ratings = df_target.mean()
    all_rating_counts = df_target.count()

    recommendations_df = pd.DataFrame(index=recommendation_indices)
    
    sims_filtered = sims.loc[recommendation_indices]
    
    # 1. Điểm tương đồng (40%) - Chuẩn hóa --- consine 0.7
    sims_min, sims_max = sims_filtered.min(), sims_filtered.max()
    recommendations_df['diem_tuong_dong'] = (sims_filtered - sims_min) / (sims_max - sims_min) if sims_max != sims_min else 0.5
    
    # 2. Điểm đánh giá trung bình (30%)
    avg_ratings_min, avg_ratings_max = all_avg_ratings.min(), all_avg_ratings.max()
    recommendations_df['diem_danh_gia'] = (all_avg_ratings - avg_ratings_min) / (avg_ratings_max - avg_ratings_min) if avg_ratings_max != avg_ratings_min else 0.5
    
    # 3. Số lượng đánh giá (30%)
    rating_counts_min, rating_counts_max = all_rating_counts.min(), all_rating_counts.max()
    recommendations_df['diem_pho_bien'] = (all_rating_counts - rating_counts_min) / (rating_counts_max - rating_counts_min) if rating_counts_max != rating_counts_min else 0.5
    
    recommendations_df = recommendations_df.fillna(0) # Đảm bảo không có NaN

    # Tính điểm tổng hợp                 ----- pearson 0.3
    recommendations_df['diem_cuoi_cung'] = (
        0.4 * recommendations_df['diem_tuong_dong'] +
        0.3 * recommendations_df['diem_danh_gia'] +
        0.3 * recommendations_df['diem_pho_bien']
    )
    
    # Sắp xếp theo final_score
    top_places = recommendations_df.sort_values('diem_cuoi_cung', ascending=False).head(top_n)
    
    recommendations = []
    for similar_place in top_places.index:
        rating_info = get_rating_summary(similar_place) 
        if rating_info:
            recommendations.append((similar_place, rating_info))
    
    print(f"[DEBUG] recommend: Trả về {len(recommendations)} gợi ý (dựa trên diem_cuoi_cung).")
    return recommendations

def ask_ollama(prompt, model="llama3", timeout=600, history=[]):
    
    context_prompt = ""
    context_turns = history[-4:] 
    
    if context_turns:
        context_prompt += "Dưới đây là lịch sử hội thoại gần nhất để bạn tham khảo ngữ cảnh:\n"
        for turn in context_turns:
            if 'user' in turn:
                context_prompt += f"Người dùng trước: {turn['user']}\n"
            if 'bot' in turn:
                context_prompt += f"Trợ lý trước: {turn['bot']}\n"
        context_prompt += "\n--- Hết lịch sử ---\n\n"

    full_prompt = context_prompt + "YÊU CẦU HIỆN TẠI:\n" + prompt
    print(f"[DEBUG] ask_ollama: Gửi Prompt (500 chars):\n{full_prompt[:500]}...")

    try:
        res = subprocess.run(
            ["ollama", "run", model, full_prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout
        )
        return res.stdout.strip() if res.stdout else "(Ollama không phản hồi)"
    except Exception as e:
        return f"(Lỗi Ollama: {e})"

def get_chatbot_response(user_input):
    user_input_original = user_input.strip() 
    if not user_input_original:
        return "Bạn chưa nhập gì cả."

    user_norm = normalize_text(user_input_original)
    print(f"[DEBUG] get_chatbot_response: Input Normalized: '{user_norm}'")

    place_found, match_type = find_place_in_input(user_norm, place_norm_map)

    if place_found:
        print("[DEBUG] get_chatbot_response: Logic Block 3 (Place Found)")
        rating_info = get_rating_summary(place_found)
        recommendations = recommend(place_found) 
        if recommendations:
            suggest_text = " - ".join([f"{p} (ĐG {r['average']}/5)" for p, r in recommendations])
            prompt = f"""
            Bạn là trợ lý du lịch chuyên nghiệp.
            Người dùng hỏi về: {place_found} (từ câu: '{user_input_original}')
            Thông tin: {rating_info['average']}/5 (từ {rating_info['count']} lượt đánh giá).

            Dựa trên thuật toán, đây là các gợi ý TƯƠNG TỰ NHẤT (BẮT BUỘC chỉ dùng danh sách này):
            {suggest_text}

            Nhiệm vụ: Trả lời ngắn gọn về {place_found} (kèm đánh giá), sau đó liệt kê các gợi ý trên. (KHÔNG DÙNG EMOJI và hoàn toàn bằng tiếng Việt).
            """
            return ask_ollama(prompt, history=[]) 
        else:
            print("[DEBUG] get_chatbot_response: Logic Block 3b (Place Found, No Recs)")
            rating_text = f"{rating_info['average']}/5 (từ {rating_info['count']} lượt đánh giá)"

            prompt = f"""
            Thông tin về {place_found}:
            Đánh giá: {rating_text}.
            
            Hiện tại tôi chưa có đủ dữ liệu để đưa ra gợi ý tương đồng cho địa điểm này. 
            Bạn có muốn hỏi thêm về giờ mở cửa, giá vé, hoặc các thông tin khác của {place_found} không? (KHÔNG DÙNG EMOJI).
            """
            # Truyền history để trả lời câu hỏi chi tiết
            return ask_ollama(prompt, history=chat_history)
#Step 2: kiểm tra từ khóa du lịch 
    
    travel_keywords = ["giới thiệu", "khuyến nghị", "địa điểm", "du lịch", "tham quan", "gợi ý", "chỗ nào", "review", "đi chơi", "đi đâu", "hcm", "saigon", "vũng tàu"]
    normalized_keywords = [normalize_text(kw) for kw in travel_keywords]
    is_travel_query = any(kw in user_norm for kw in normalized_keywords)
    print(f"[DEBUG] get_chatbot_response: Is Travel Query (Keyword Match): {is_travel_query}")

    if is_travel_query:
        print("[DEBUG] get_chatbot_response: Logic Block 2 (General Travel Query)")
        # Gợi ý địa điểm phổ biến
        popular_places = df.columns.tolist()[:5]
        place_list = ", ".join(popular_places)
        prompt = f"""Người dùng hỏi chung chung về du lịch: '{user_input_original}'.
        Hãy trả lời bằng cách thân thiện gợi ý 3-5 địa điểm phổ biến nhất (ví dụ: {place_list}).
        Sau đó mời họ hỏi chi tiết về một trong các địa điểm đó. (KHÔNG DÙNG EMOJI)."""
        return ask_ollama(prompt, history=chat_history)

#KHÔNG TÌM THẤY ĐỊA ĐIỂM VÀ KHÔNG CÓ TỪ KHÓA DU LỊCH
    print("[DEBUG] get_chatbot_response: Logic Block 1 (General Conversation)")
    prompt = f"""Người dùng nói: '{user_input_original}'.
    Bạn là trợ lý AI thân thiện, hãy trả lời tự nhiên bằng tiếng Việt. (KHÔNG DÙNG EMOJI)."""
    return ask_ollama(prompt, history=chat_history)
@app.route("/")
def index():
    return render_template("index.html", history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("message", "").strip()
    bot_response = get_chatbot_response(user_input)
    chat_history.append({"user": user_input, "bot": bot_response})
    save_history(chat_history)
    return jsonify({"response": bot_response})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    global chat_history
    chat_history = []
    save_history(chat_history)
    return jsonify({"message": "Đã xoá toàn bộ lịch sử hội thoại."})

if __name__ == "__main__":
    if not df.empty:
        print("Dữ liệu đã tải, số địa điểm:", len(df.columns))
    else:
        print("Lỗi: Không có dữ liệu")
    app.run(debug=True)
