# 🛍️ Shopping Assistant Torob

A powerful, multilingual shopping assistant API built with **FastAPI**, supporting Persian product queries, natural language understanding, 
and product intelligence features like product extraction, image classification, and price comparison.

---

## 🚀 Features

* 🔍 **Text Query Parsing** (Persian): Extracts product names from user text using rules + LLM (OpenAI).
* 🖼️ **Image Classification**: Detects product category from uploaded images (Scenario 6/7).
* 🔁 **Product Comparison**: Supports comparing multiple product options.
* 📊 **Shop and Warranty Stats**: Computes number of shops with/without warranty for a product.
* 🏷️ **Price Stats**: Retrieves minimum, average, or maximum product prices.
* 💬 **Structured Chat API**: Handles `ChatRequest` / `ChatResponse` payloads.

---

## ⚠️ Current Status

⚠️ **Codebase Status:** The project works well, but the code is not fully modular. The logic in `main.py` and `services.py` should be
**further modularized** into smaller, testable components (e.g., intent detection, response generation, and classification pipelines) to improve maintainability and readability.

---

## 🧰 Tech Stack

* **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
* **Data**: Parquet (via Pandas), JSON
* **AI/ML**: OpenAI (via `openai.AsyncOpenAI`)
* **Image Processing**: Pillow
* **Text Normalization**: Hazm
* **Similarity Matching**: RapidFuzz

---

## 📂 Project Structure

```bash
.
├── app/
│   ├── main.py                
│   ├── services.py            
│   ├── schemas.py            
│   ├── dependencies.py        
│   ├── config.py             
├── requirements.txt          
├── Dockerfile                
├── docker-compose.yml        
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-org/shop_assistant_torob.git
cd shop_assistant_torob
```

### 2. Environment Setup

Create a `.env` file with the following variables:

```env
# .env
DATA_FILE_PATH=/data/products.parquet
MEMBERS_FILE_PATH=/data/members.parquet
SHOPS_FILE_PATH=/data/shops.parquet
CITIES_FILE_PATH=/data/cities.parquet
CATEGORIES_FILE_PATH=/data/categories.parquet
SEARCHES_FILE_PATH=/data/searches.parquet

OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or run with Docker:

```bash
docker build -t shopping-assistant .
docker run -p 8080:8080 shopping-assistant
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

Server starts at: [http://localhost:8080](http://localhost:8080)

---

## 📬 API Endpoints


### `POST /chat`

Main endpoint to process user queries (text or image).

**Request Body (`ChatRequest`)**:

```json
{
  "chat_id": "abc123",
  "messages": [
    {"type": "text", "content": "قیمت میز چوبی چقدره؟"}
  ]
}
```

**Response (`ChatResponse`)**:

```json
{
  "message": "۱۵۰۰۰۰۰ تومان"
  "base_random_keys": null,
  "member_random_keys": null
}
```

---

## 📸 Supported Query Scenarios

| Scenario   | Description                                         |
| ---------- | --------------------------------------------------- |
| Scenario 0 | Ping test, returns base/member key                  |
| Scenario 1 | Product name → Key resolution                       |
| Scenario 2,3 | Attribute Q&A from features (e.g., "چند کشو داره؟") |
| Scenario 5 | Product comparison                                  |
| Scenario 6 | Image classification                                |
| Scenario 7 | Image → Best matching product                       |

---

## 🐳 Docker Compose (Optional)

```yaml
# docker-compose.yml
version: '3.9'
services:
  api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
    env_file:
      - .env
```

Then:

```bash
docker-compose up
```

---

## 🧠 AI 

* Uses OpenAI's GPT for:

  * Product name extraction
  * Attribute value detection
  * Image classification support
* Uses Persian text normalization (Hazm-style)
* Fuzzy string matching for product resolution using RapidFuzz


## ✅ TODO 

### 🧱 Code Refactoring

* [ ] **Modularize** the core logic in `main.py` into smaller services (e.g., intent detection, response handling).
* [ ] Split `services.py` into multiple focused modules.
* [ ] Add proper unit tests and integration tests.

### 🧪 Quality & Maintenance

* [ ] Add type hints and docstrings throughout the codebase.
* [ ] Improve error handling and validation.




