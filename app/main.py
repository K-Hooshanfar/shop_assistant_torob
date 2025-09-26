# app/main.py
import logging
import re
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Depends
from pathlib import Path
from datetime import datetime, timezone
import aiofiles
import json

from .schemas import ChatRequest, ChatResponse
from .dependencies import get_products_df, get_members_df, get_shops_df, get_cities_df, get_categories_df
from . import services

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG_DIR = Path(__file__).resolve().parent / "logs"
REQUESTS_DIR = LOG_DIR / "requests"


async def _save_payload(payload: ChatRequest) -> Optional[Path]:

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc)
        ts_slug = ts.strftime("%Y%m%dT%H%M%S%fZ")

        try:
            data = payload.model_dump()
        except Exception:
            data = payload.dict()

        chat_id = str(getattr(payload, "chat_id", "nochat"))
        safe_cid = re.sub(r"[^A-Za-z0-9._-]+", "_", chat_id)[:60]

        per_request_path = REQUESTS_DIR / f"{ts_slug}_{safe_cid}.json"
        async with aiofiles.open(per_request_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

        ndjson_path = LOG_DIR / "requests.ndjson"
        line_obj = {"_ts": ts.isoformat(), **data}
        async with aiofiles.open(ndjson_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

        return per_request_path
    except Exception:
        logging.exception("Failed to persist payload")
        return None


app = FastAPI(title="Shopping Assistant", version="1.4")

@app.get("/")
def healthcheck():
    return {"status": "ok"}


def _latest_text_message(req: ChatRequest) -> Optional[str]:
    for m in reversed(req.messages or []):
        if m.type and m.type.lower() == "text" and m.content and m.content.strip():
            return m.content.strip()
    return None

def _latest_image_data_url(req: ChatRequest) -> Optional[str]:

    for m in reversed(req.messages or []):
        if m.type and m.type.lower() == "image" and m.content and m.content.strip():
            c = m.content.strip()
            if c.startswith("data:image/"):
                return c
    return None

# ---------- Scenario 0 helpers ----------
_BASE_CMD_RE   = re.compile(r"^return\s+base\s+random\s+key:\s*(\S+)\s*$", re.IGNORECASE)
_MEMBER_CMD_RE = re.compile(r"^return\s+member\s+random\s+key:\s*(\S+)\s*$", re.IGNORECASE)

def _scenario0_response(user_text: Optional[str]) -> Optional[ChatResponse]:
    """
    Sanity-check commands (short-circuit): ping/pong, return base/member key.
    """
    if not user_text:
        return None

    text = user_text.strip()

    # 0.a ping → pong
    if text.lower() == "ping":
        logging.info("✅ Scenario 0: ping -> pong")
        return ChatResponse(message="pong")

    # 0.b return base random key: <id>
    m_base = _BASE_CMD_RE.match(text)
    if m_base:
        key = m_base.group(1)
        logging.info(f"✅ Scenario 0: return base key -> {key}")
        return ChatResponse(base_random_keys=[key])

    # 0.c return member random key: <id>
    m_member = _MEMBER_CMD_RE.match(text)
    if m_member:
        key = m_member.group(1)
        logging.info(f"✅ Scenario 0: return member key -> {key}")
        return ChatResponse(member_random_keys=[key])

    return None
# ---------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    products_df: pd.DataFrame = Depends(get_products_df),
    members_df: pd.DataFrame = Depends(get_members_df),
    shops_df: pd.DataFrame = Depends(get_shops_df),           # ← inject shops_df
    cities_df: pd.DataFrame = Depends(get_cities_df),
    categories_df: pd.DataFrame = Depends(get_categories_df),
) -> ChatResponse:
    try:
        await _save_payload(payload)

        user_text = _latest_text_message(payload)
        if not user_text:
            logging.warning("No text message found in the request.")
            return ChatResponse()

        logging.info(f"--- Processing Query: '{user_text}' ---")

        # Scenario 0 (ping / explicit key returns) ...
        s0 = _scenario0_response(user_text)
        if s0 is not None:
            return s0

        # Scenario 7 - map image -> best base product (returns base_random_keys only)
        try:
            if services.is_image_to_base_query(user_text):
                try:
                    query = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
                except Exception:
                    query = {"messages": [{"type": "text", "content": user_text}]}

                labels = await services.classify_main_object_from_image_query(query)
                if labels:
                    key = services.choose_base_by_labels(
                        labels,
                        categories_df,
                        products_df,
                        keylist_path=getattr(services.settings, "scenario7_keylist_path", "label.txt"),
                        strict_keylist=False,
                    )
                    if key:
                        logging.info(f"✅ Scenario 7 DETECTED: labels={labels} -> base_key={key}")
                        return ChatResponse(base_random_keys=[key])
                logging.warning("Scenario 7: could not resolve a base key from labels/category/keylist.")
        except Exception as e:
            logging.exception(f"Scenario 7 failed: {e}")

        # Scenario 6 - Main object in an image
        try:
            if services.is_main_object_query(user_text):
                try:
                    query = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
                except Exception:
                    query = {"messages": [{"type": "text", "content": user_text}]}

                labels = await services.classify_main_object_from_image_query(query)
                if labels and labels.get("coarse_label"):
                    coarse = labels["coarse_label"]
                    fine = labels.get("fine_label") or "unknown"
                    logging.info(f"✅ Scenario 6 DETECTED: main object in image -> {coarse} / {fine}")
                    return ChatResponse(message=f"{coarse}, {fine}")

                logging.warning("Scenario 6: no usable text/image content or classifier output.")
        except Exception as e:
            logging.exception(f"Scenario 6 failed: {e}")

        # Scenario 5 - Product Comparison (check first, before single product extraction)
        comparison_result = await services.handle_product_comparison(
            user_text, products_df, members_df, shops_df, cities_df
        )
        if comparison_result:
            logging.info("✅ Scenario 5 DETECTED: Product comparison (price/feature/coverage)")
            return comparison_result

        # Step 1: product name extraction
        extracted_name = await services.extract_product_name_from_query(user_text)
        if not extracted_name:
            logging.warning("LLM could not extract a product name.")
            return ChatResponse()
        logging.info(f"🤖 LLM extracted product name: '{extracted_name}'")


        # Step 3: specific base lookup if it wasn't a category name
        best_key_list = services.find_product_key_in_df(extracted_name, products_df)
        if not best_key_list:
            logging.warning(f"Product '{extracted_name}' not found (category + base lookup).")
            return ChatResponse()
        best_key = str(best_key_list[0])
        product_row = products_df[products_df['random_key'] == best_key].iloc[0]
        logging.info(f"✅ Scenario 1 SUCCEEDED: Found product '{product_row.get('persian_name', best_key)}' (Key: {best_key})")

        # 1) warranty count
        if services.is_warranty_count_query(user_text):
            logging.info("Intent detected: warranty shop-count")
            try:
                cnt = services.get_warranty_shop_count_for_base(best_key, members_df, shops_df)
                safe_cnt = int(cnt or 0)
                logging.info(
                    "Warranty shop-count computed: count=%d (base_key=%s)",
                    safe_cnt, best_key,
                    extra={"base_key": best_key, "count": safe_cnt, "intent": "warranty_shop_count"}
                )
                return ChatResponse(message=str(safe_cnt))
            except Exception as e:
                logging.exception(f"Failed to compute warranty shop-count for base_key={best_key}: {e}")
                return ChatResponse(message="0")

        # 2) plain shop count (no warranty)
        if services.is_shop_count_query(user_text):
            logging.info("Intent: plain shop-count (base_key=%s)", best_key)
            city = services.extract_city_from_query(user_text, cities_df)
            if city:
                logging.info("Intent: plain shop-count (base_key=%s, city=%s)", best_key, city)
                cnt = services.get_shop_count_for_base(
                    best_key,
                    members_df,
                    shops_df,
                    cities_df,
                    city=city,
                )
            else:
                logging.info("Intent: plain shop-count (base_key=%s, city=ALL)", best_key)
                cnt = services.get_shop_count_for_base(
                    best_key,
                    members_df,
                    shops_df,
                    cities_df,
                    city=None,
                )
            safe_cnt = int(cnt or 0)
            logging.info(
                "Plain shop-count result: count=%d (base_key=%s, city=%s)",
                safe_cnt, best_key, city or "ALL"
            )
            return ChatResponse(message=str(safe_cnt))

        # Scenario 3: price query
        if services.is_price_query(user_text):
            kind = services.detect_price_type(user_text)
            city = services.extract_city_from_query(user_text, cities_df)
            agg = "per_offer" if kind == "avg" else "per_shop_min"
            stats = services.compute_price_stats_for_base(
                best_key, members_df, shops_df, city=city, cities_df=cities_df, aggregate=agg
            )
            if not stats:
                logging.warning("Price requested but no member prices found for this base.")
                return ChatResponse()

            val = stats.get(kind) or stats.get("min")
            if val is None:
                return ChatResponse()

            logging.info(f"✅ Scenario 3 DETECTED: price intent='{kind}' city='{city}' -> {val}")
            return ChatResponse(message=str(val))

        # Scenario 2: attribute Q&A
        product_features = services.parse_extra_features(product_row.get('extra_features', {}))
        requested_attribute = await services.extract_attribute_via_llm(user_text, product_features)

        if requested_attribute:
            logging.info(f"✅ Scenario 2 DETECTED: '{requested_attribute}'")
            value = services.get_feature_value(product_features, requested_attribute)
            if value is None:
                logging.warning(f"Attribute '{requested_attribute}' not found for key={best_key}.")
                return ChatResponse()
            message = await services.attribute_answer_via_llm(requested_attribute, value, user_text)
            return ChatResponse(message=message or value)

        try:
            s = services.normalize_fa(extracted_name or "")

            is_specific = bool(
                services.extract_model_token(s)
                or services.extract_catalog_code_any(s)
                or re.search(r"[A-Za-z]", s)
            )

            if not is_specific:
                generic_pats = [
                    r"(?<!\S)چند\s*کاره(?!\S)",
                    r"(?<!\S)خوب(?!\S)",
                    r"(?<!\S)با\s*کیفیت(?!\S)",
                    r"(?<!\S)کیفیت\s*بالا(?!\S)",
                    r"(?<!\S)قیمت\s*مناسب(?!\S)",
                    r"(?<!\S)به\s*صرفه(?!\S)",
                ]
                s_stripped = s
                for pat in generic_pats:
                    s_stripped = re.sub(pat, "", s_stripped, flags=re.IGNORECASE)
                s_stripped = re.sub(r"\s+", " ", s_stripped).strip() or s

                for candidate in (s_stripped, s):
                    cat_id = services._pick_category_id_from_name_exact(candidate, categories_df)
                    if cat_id is not None:
                        base_key = services._pick_one_base_key_for_category(cat_id, products_df)
                        if base_key:
                            logging.info(
                                f"✅ Category-first (exact, generic): '{extracted_name}' -> '{candidate}' -> category_id={cat_id} -> base_key={base_key}"
                            )
                            return ChatResponse(base_random_keys=[str(base_key)])
        except Exception as e:
            logging.info(f"Category-first (exact) path skipped due to error: {e}")

        logging.info("❌ No attribute/price/warranty intent: returning base key.")
        return ChatResponse(base_random_keys=[best_key])

    except Exception as ex:
        logging.exception(f"/chat handler failed: {ex}")
        return ChatResponse()

