# app/services.py
import logging
import re
import json
import ast
from typing import Optional, List, Dict, Any, Union
from .schemas import ChatResponse
import pandas as pd
import base64, io, urllib.parse
from PIL import Image
import base64
import logging


try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

from .config import settings


client: Optional["AsyncOpenAI"] = None
try:
    if AsyncOpenAI is not None:
        kwargs = {}
        if getattr(settings, "openai_api_key", None):
            kwargs["api_key"] = settings.openai_api_key
        if getattr(settings, "openai_base_url", None):
            kwargs["base_url"] = settings.openai_base_url
        if kwargs:
            client = AsyncOpenAI(**kwargs)
except Exception as e:
    logging.info(f"OpenAI client init skipped: {e}")
    client = None

SYSTEM_PROMPT = (
    "You are a product-name extractor. Your ONLY output must be a single, final, canonical product name as plain text.\n"
    "No explanations, no JSON, no quotes, no extra words.\n"
    "\n"
    "General Rules:\n"
    "1) Output ONLY the final product name on one line.\n"
    "2) Remove greetings/politeness and generic request/availability chatter, but KEEP domain-specific tokens that are part of catalog naming.\n"
    "3) Remove parenthetical/bracketed side notes not part of the title (e.g. payment/shipping like «پس کرایه»), unless they contain a true «کد …».\n"
    "3.1) SHIPPING/AVAILABILITY — DO NOT DROP coverage phrases like:\n"
    " {\"ارسال به سراسر ایران\", \"ارسال به سراسر کشور\", \"امکان ارسال به سراسر ایران\", \"امکان ارسال به سراسر کشور\", \"ارسال به تمام نقاط ایران\", \"ارسال به تمام ایران\"}.\n"
    " If any variant appears anywhere (even inside a relative clause like \"... که امکان ارسال به سراسر ایران دارد\"),\n"
    " extract the shortest canonical form (prefer \"ارسال به سراسر ایران\"; if only \"کشور\" appears, use \"ارسال به سراسر کشور\"),\n"
    "4) Keep «کد …» if it is a catalog code. Keep «مدل …» only if it looks like a true catalog title (e.g. «صندلی اداری مدل K12»). Drop generic numeric variants like «مدل 122» after a generic clock.\n"
    "5) If the text contains an inclusion list with «شامل …», compose the final name as «<base> <item1> و <item2> و …». Split on commas and «و».\n"
    "6) Normalize whitespace; remove ZWNJ; keep a clean single-spaced result.\n"
    "7) If nothing meaningful can be extracted, return an empty string.\n"
    "8) Parentheses/brackets: Remove shipping/payment side notes EXCEPT a small keep-list that must be preserved verbatim if present:\n"
    "   keep-list = {\"پس کرایه\", \"پرداخت درب منزل\", \"ارسال رایگان\"}.\n"
    "   If any of these appear in ()[]{} at the end, KEEP the whole parenthetical, unchanged, at the end of the title.\n"
    "\n"
    "NUMBERS, UNITS, COUNTS:\n"
    "9) Preserve explicit size/height/width values (e.g., «سایز ۶۰», «ارتفاع ۲۵ سانتی‌متر»). Normalize «ارتفاع N سانتی‌متر» → «N سانتی».\n"
    "10) Never invent numbers. Always use what the user wrote. If a token contains LATIN/English letters (model/catalog codes like D104, VC750), keep its digits in ASCII (0-9), e.g., «D104», not «D۱۰۴».\n"
    "\n"
    "CATEGORY ANCHORS (IMPORTANT):\n"
    "11) The final product name must contain the same category anchor noun the user used (e.g., «شمع», «ساعت», «میز», «کفش», …). If you cannot detect the same anchor, return empty.\n"
    "\n"
    "INTENT HANDLING:\n"
    "A) If the user asks a question about an attribute (cues like «چه/چقدر/چند/چیست/؟/آیا»), DO NOT answer. Just extract the product title.\n"
    "B) If the user is selecting/commanding (e.g. «می‌خواهم»، «موجود کنید»), still extract the canonical title.\n"
    "C) Never return anything except the one-line product name (or empty).\n"
    "\n"
    "Examples (input → output):\n"
    "«سلام، من به دنبال خرید ساعت دیواری مدرن فلزی سایز 60 مدل 122 هستم. (پس کرایه)» → «ساعت دیواری مدرن فلزی سایز ۶۰ سانتی (پس کرایه)»\n"
    "«ست گلدان رومیزی شامل سفالی کوچک، متوسط، بزرگ» → «ست گلدان رومیزی سفالی کوچک و متوسط و بزرگ»\n"
    "«شمع تزیینی شمعدونی پیچی سبز با ارتفاع ۲۵ سانتی‌متر که ۱ عددی است را می‌خواهم.» → «شمع تزیینی شمعدونی پیچی سبز ۲۵ سانتی»\n"
    "«من دنبال فرشینه مخمل با ترمزگیر و عرض ۱ متر، طرح آشپزخانه با کد ۰۴ هستم.» → «فرشینه مخمل با ترمزگیر عرض 1 متر طرح آشپزخانه کد 04»\n"
    "\n"
    "Attribute-question examples:\n"
    "«عرض پارچه مخمل مدل رویال چقدره؟» → «پارچه مخمل مدل رویال»\n"
    "«توان جاروبرقی مدل VC750 چقدره؟» → «جاروبرقی مدل VC750»\n"
    "«اصالت کفش ورزشی آلفا پرو رو تأیید می‌کنید؟» → «کفش ورزشی آلفا پرو»\n"
    "\n"
    "If no product identified:\n"
    "«جنسش چیه؟» → «»\n"
    "\n"
    "Reminder: Return ONLY the final one-line product name. No explanations.\n"
)


def _normalize_persian_numerals(text: str) -> str:
    if not text:
        return ""
    numeral_map = {
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    return "".join(numeral_map.get(ch, ch) for ch in text)


def to_persian_digits(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    trans = str.maketrans({
        '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
        '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
        '.': '٫',
    })
    return text.translate(trans)


_DIACRITICS_RE = re.compile(r'[\u064B-\u065F\u0670]')


def normalize_fa(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _normalize_persian_numerals(text)
    trans = str.maketrans({
        'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و', 'ئ': 'ی', 'ۀ': 'ه', 'ٰ': '',
        '‌': ' ', '\u200c': ' ', '\u200f': '', '\u200e': '', '\u061C': '',
        '٬': ',', '،': ',', '؛': ';', 'ـ': '',
        '«': '"', '»': '"', '–': '-', '−': '-',
        '[': '(', ']': ')', '{': '(', '}': ')',
    })
    text = text.translate(trans)
    text = _DIACRITICS_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_and_normalize_string(text: str) -> str:
    return normalize_fa(text)


_CODE_RE = re.compile(r'\((?:کد|كد)\s*[:\-]?\s*([A-Za-z0-9\-]+)\)')


def extract_code_token(text: str) -> Optional[str]:
    if not text:
        return None
    m = _CODE_RE.search(text)
    return m.group(1) if m else None


def _ensure_norm_column(df: pd.DataFrame) -> None:
    if 'persian_name' in df.columns and '_norm_persian_name' not in df.columns:
        df['_norm_persian_name'] = df['persian_name'].astype(str).map(normalize_fa)


_INCLUSION_RE = re.compile(
    r'(?:^|[\s"“])(?:درخواست\s+)?(?:محصول\s+)?(?P<base>[\u0600-\u06FFA-Za-z0-9\- ]+?)\s+شامل\s+(?P<list>[^\.؟!\n\r]+)',
    re.IGNORECASE,
)


def _normalize_spacing(s: str) -> str:
    s = (s or "").replace("\u200c", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _compose_name_from_inclusions(query: str) -> Optional[str]:
    norm = _normalize_spacing(query or "")
    m = _INCLUSION_RE.search(norm)
    if not m:
        return None

    base = m.group('base').strip()
    rest = m.group('list').strip()

    rest = re.sub(r'[\.؟!]+$', '', rest).strip()

    parts = re.split(r'\s*[,،]\s*|\s+و\s+', rest)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return None

    return f"{base} {' و '.join(parts)}"



_PERSIAN_TO_ASCII_DIGITS = str.maketrans({
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
})
_LATIN_RE = re.compile(r'[A-Za-z]')


def enforce_ascii_digits_in_latin_tokens(name: str) -> str:
    if not name:
        return name
    parts = name.split()
    fixed = []
    for tok in parts:
        if _LATIN_RE.search(tok):
            fixed.append(tok.translate(_PERSIAN_TO_ASCII_DIGITS))
        else:
            fixed.append(tok)
    return " ".join(fixed)



async def extract_product_name_from_query(query: str) -> Optional[str]:

    composite = _compose_name_from_inclusions(query)
    if composite:
        out = composite
        return enforce_ascii_digits_in_latin_tokens(out)

    # LLM path
    try:
        model = getattr(settings, "openai_model", None)
        local_client = AsyncOpenAI() if (AsyncOpenAI and model) else None
        if local_client is not None and model:
            completion = await local_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            name = completion.choices[0].message.content if completion.choices else None

            composite_llm = _compose_name_from_inclusions(query)
            out = (composite_llm or (name.strip() if name else None))
            if out:
                out = enforce_ascii_digits_in_latin_tokens(out)
            return out
    except Exception as e:
        logging.info(f"LLM extraction failed; will try heuristic. Error: {e}")

    try:
        if "(کد" in query or "(كد" in query:
            m = re.search(r'(.{0,40})\s*(\((?:کد|كد)[^)]+\))', query)
            if m:
                left = re.sub(r'\b(را|برای|يك|یک|ميخواهم|می‌خواهم|میخواهم|لطفاً|لطفا)\b$', '', m.group(1).strip())
                out = f"{left} {m.group(2).strip()}".strip()
                return enforce_ascii_digits_in_latin_tokens(out)

        composite = _compose_name_from_inclusions(query)
        if composite:
            return enforce_ascii_digits_in_latin_tokens(composite)

        out = _normalize_spacing(query) or None
        if out:
            out = enforce_ascii_digits_in_latin_tokens(out)
        return out
    except Exception:
        composite = _compose_name_from_inclusions(query)
        out = composite or (_normalize_spacing(query) or None)
        return enforce_ascii_digits_in_latin_tokens(out) if out else None


_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ASCII_DIGITS = "0123456789"


def _flex_code_pattern(code: str) -> str:

    code = str(code or "")
    parts: List[str] = []
    for ch in code:
        if ch.isdigit():
            if ch in _ASCII_DIGITS:
                pers = _PERSIAN_DIGITS[_ASCII_DIGITS.index(ch)]
                parts.append(f"[{re.escape(ch)}{re.escape(pers)}]")
            else:
                parts.append(re.escape(ch))
        elif ch.isalpha():
            parts.append(re.escape(ch))
        else:
            parts.append(r"[\s\-]*")
        parts.append(r"[\s\-]*")
    pat = "".join(parts)
    pat = re.sub(r"(?:\[\\s\\\-]\*)+$", "", pat)
    return pat


_MODEL_RE = re.compile(r'مدل\s+([A-Za-z][A-Za-z0-9\-_]+)', re.IGNORECASE)

def extract_model_token(text: str) -> Optional[str]:
    m = _MODEL_RE.search(normalize_fa(text or ""))
    return m.group(1) if m else None


_STOPWORDS = {"مدل", "طرح", "پایه", "سایز", "اندازه", "کد", "کيت", "کیفیت", "اصل", "اورجینال"}


def _loosy_token_pattern(tok: str) -> str:
    letters = list(tok)
    return r"\s*".join(map(re.escape, letters))


def _tokenize_fa(s: str) -> List[str]:
    s = normalize_fa(s)
    toks = re.split(r"[\s,،\-_/()+]+", s)
    toks = [t for t in toks if t and t not in _STOPWORDS]
    return toks


_COUNT_RE = re.compile(r'(\d+)\s*(?:عدد(?:ی)?|تایی|بسته)')


def _extract_count_token(s: str) -> Optional[str]:
    s = normalize_fa(s or "")
    m = _COUNT_RE.search(s)
    return m.group(1) if m else None


_CODE_ANY_RE = re.compile(
    r'(?:^|[\s\(\[\{])(?:کد|كد)\s*[:\-]?\s*([A-Za-z0-9\-]+)\b',
    re.IGNORECASE
)


def extract_catalog_code_any(text: str) -> Optional[str]:
    if not text:
        return None
    m = _CODE_ANY_RE.search(normalize_fa(text))
    return m.group(1) if m else None


_UNITS = {
    "متر", "سانتی", "سانتی‌متر", "سانتی متر", "سانت", "میلی", "میلی‌متر", "میلیمتر",
    "گرم", "کیلو", "کیلوگرم", "لیتر", "اینچ", "inch", "cm", "mm"
}
_FILLERS = {
    "با", "و", "در", "از", "به", "برای", "روی", "تا",
    "محصول", "کالا", "شامل", "ست", "بسته", "عدد", "تایی",
    "طول", "عرض", "ارتفاع", "ابعاد", "اندازه"
}
_GENERIC_TOKENS = _STOPWORDS | _UNITS | _FILLERS


def _significant_tokens(s: str) -> List[str]:
    toks = _tokenize_fa(s)
    return [t for t in toks if t not in _GENERIC_TOKENS and len(t) >= 2]


def find_product_key_in_df(product_name: str, df: pd.DataFrame, *, max_candidates: int = 1) -> Optional[List[str]]:
    try:
        if df is None or df.empty:
            return None
        _ensure_norm_column(df)

        q_norm = clean_and_normalize_string(product_name)
        if not q_norm:
            return None

        catalog_code = extract_catalog_code_any(product_name)
        model_code = extract_model_token(product_name)
        if not model_code:
            m = re.search(r'\b([A-Za-z]+[A-Za-z0-9\-]*\d+[A-Za-z0-9\-]*)\b', q_norm)
            model_code = m.group(1) if m else None

        candidate_df = df

        if catalog_code:
            flex_pat = _flex_code_pattern(str(catalog_code))
            mask_code_any = candidate_df['_norm_persian_name'].str.contains(
                flex_pat, regex=True, case=False, na=False
            )
            if mask_code_any.any():
                candidate_df = candidate_df.loc[mask_code_any].copy()

        if model_code and not candidate_df.empty:
            flex_pat = _flex_code_pattern(model_code)
            mask_code = candidate_df['_norm_persian_name'].str.contains(flex_pat, regex=True, case=False, na=False)
            if mask_code.any():
                candidate_df = candidate_df.loc[mask_code].copy()

        q_count = _extract_count_token(product_name)
        if q_count and not candidate_df.empty:
            mask_count = candidate_df['_norm_persian_name'].str.contains(
                rf'\b{re.escape(q_count)}\s*(?:عدد(?:ی)?|تایی|بسته)\b', regex=True, na=False
            )
            if mask_count.any():
                candidate_df = candidate_df.loc[mask_count] if not candidate_df.empty else df.loc[mask_count]
                candidate_df = candidate_df.copy()

        mask_eq = (candidate_df['_norm_persian_name'] == q_norm)
        if mask_eq.any():
            return [str(candidate_df.loc[mask_eq, 'random_key'].iloc[0])]

        sig_toks = _significant_tokens(q_norm)
        if sig_toks and not candidate_df.empty:
            sig_or_pat = "|".join(_loosy_token_pattern(t) for t in set(sig_toks))
            mask_sig_any = candidate_df['_norm_persian_name'].str.contains(sig_or_pat, regex=True, na=False)
            if mask_sig_any.any():
                candidate_df = candidate_df.loc[mask_sig_any].copy()

        toks = _tokenize_fa(q_norm)
        key_toks = [t for t in toks if len(t) >= 2 and t not in _GENERIC_TOKENS]
        if model_code:
            key_toks.append(model_code)

        work_df = candidate_df
        for t in key_toks:
            pat = _loosy_token_pattern(t)
            work_df = work_df.loc[work_df['_norm_persian_name'].str.contains(pat, regex=True, na=False)]
            if work_df.empty:
                break
        if not work_df.empty:
            return [str(work_df['random_key'].iloc[0])]

        mask_sub = candidate_df['_norm_persian_name'].str.contains(q_norm, na=False, regex=False)
        if mask_sub.any():
            return [str(candidate_df.loc[mask_sub, 'random_key'].iloc[0])]

        q_wo_paren = re.sub(r'\([^)]*\)', '', q_norm).strip()
        if q_wo_paren and q_wo_paren != q_norm:
            mask_sub2 = candidate_df['_norm_persian_name'].str.contains(q_wo_paren, na=False, regex=False)
            if mask_sub2.any():
                return [str(candidate_df.loc[mask_sub2, 'random_key'].iloc[0])]

        try:
            from rapidfuzz import process, fuzz
            names = candidate_df['_norm_persian_name'].astype(str).tolist()

            long_enough = [i for i, n in enumerate(names) if len(n) >= 4]

            def _overlap_ok(name: str) -> bool:
                if not sig_toks:
                    return True
                count = 0
                for t in sig_toks:
                    if re.search(_loosy_token_pattern(t), name, flags=re.UNICODE):
                        count += 1
                        if count >= 2:
                            return True
                return False

            indices = [i for i in long_enough if _overlap_ok(names[i])]
            if indices:
                sub_names = [names[i] for i in indices]
                sub_df = candidate_df.iloc[indices]

                best = process.extractOne(q_norm, sub_names, scorer=fuzz.token_set_ratio)
                if best and best[1] >= 82:
                    idx = indices[best[2]]
                    return [str(sub_df.iloc[best[2]]['random_key'])]

                best2 = process.extractOne(q_norm, sub_names, scorer=fuzz.partial_ratio)
                if best2 and best2[1] >= 90:
                    idx = indices[best2[2]]
                    return [str(sub_df.iloc[best2[2]]['random_key'])]
        except Exception as e:
            logging.info(f"Fuzzy fallback unavailable/skipped: {e}")

        return None
    except Exception as ex:
        logging.exception(f"find_product_key_in_df failed: {ex}")
        return None


def parse_extra_features(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    return {}


# -------------------- Scenario 2: attribute Q&A --------------------

ATTRIBUTE_SEARCH_TERMS: Dict[str, List[str]] = {
    "width": ["عرض", "عرض پارچه", "پهنا", "پهنای", "پهن"],
    "meterage": ["متراژ", "طول پارچه", "طول رول", "طول طاقه"],
    "stock_status": ["موجودی", "وضعیت موجودی", "وضعیت", "نو", "آکبند", "کارکرده", "دست دوم", "استوک"],
    "originality": ["اصالت", "اصل بودن", "اورجینال", "اصل"],
    "body_material": ["جنس بدنه", "جنس"],
    "number_drawers": ["تعداد کشو", "کشو"],
    "number_pages": ["تعداد صفحات", "تعداد صفحه"],
    "power": ["توان", "قدرت"],
    "usage": ["کاربرد", "موارد مصرف"],
}

_AVAIL_CHATTER_RE = re.compile(
    r"(?:اگر|اگه)\s+(?:موجود|داشت(?:ه)?|دار)ه?ید",
    re.IGNORECASE,
)


def _is_availability_chatter(text: str) -> bool:
    return bool(_AVAIL_CHATTER_RE.search(normalize_fa(text or "")))


async def extract_attribute_via_llm(
        query: str,
        product_features: Dict[str, Any]
) -> Optional[str]:

    if _is_availability_chatter(query):
        return None
    try:
        model = getattr(settings, "openai_model", None)
        if client is None or not model:
            return None

        available_keys = list(product_features.keys())
        if not available_keys:
            return None

        attribute_mapping = {
            key: ATTRIBUTE_SEARCH_TERMS.get(key, [key])
            for key in available_keys
        }

        system_prompt = (
            "You are an expert attribute classifier. Your task is to determine if the user is asking a DIRECT QUESTION about a product's attributes.\n\n"
            "Instructions:\n"
            "1. Analyze the user's query to understand their INTENT. Are they asking for information (e.g., 'What is the width?', 'How many drawers?') or are they selecting/commanding a product (e.g., 'I want the four-drawer dresser', 'Get me this one')?\n"
            "2. Only classify an attribute if the query is a clear and direct QUESTION. Look for question words (like چقدر, چیست, چه, چند) or a question mark (?).\n"
            "3. If the query is a command, a statement, or simply names the product, it is NOT an attribute question. In this case, you MUST respond with 'N/A'.\n"
            "4. If it is a clear question about an attribute from the mapping, respond with its canonical English key. Otherwise, respond with 'N/A'."
        )

        user_prompt = (
            f"User Query: \"{query}\"\n\n"
            f"Attribute Mapping:\n{json.dumps(attribute_mapping, ensure_ascii=False)}"
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        result = completion.choices[0].message.content if completion.choices else None

        if result and result.strip().lower() != 'n/a' and result.strip() in attribute_mapping:
            return result.strip()

        return None
    except Exception as e:
        logging.info(f"LLM attribute extraction failed: {e}")
        return None


def get_feature_value(features: Dict[str, Any], attr: str) -> Optional[str]:
    if not features:
        return None
    key_map = {str(k).strip().lower(): v for k, v in features.items()}
    attr_lower = attr.lower()
    if attr_lower in key_map:
        value = key_map[attr_lower]
        return str(value).strip() if value is not None else None
    return None


async def attribute_answer_via_llm(attr: str, value: str, query: str) -> Optional[str]:
    try:
        model = getattr(settings, "openai_model", None)
        if client is None or not model:
            return None
        sys = (
            "You are a formatter that returns ONLY the attribute VALUE.\n"
            "\n"
            "Input you receive:\n"
            "- The user's Persian query (context only)\n"
            "- The attribute key being asked about (context only)\n"
            "- The attribute VALUE string\n"
            "\n"
            "Your job:\n"
            "1) Output ONLY the VALUE string, with minimal normalization.\n"
            "2) Do NOT add words, units, labels, or explanations. No punctuation unless it exists inside the value.\n"
            "3) Normalization rules:\n"
            "   - Trim leading/trailing whitespace and surrounding quotes.\n"
            "   - If the value contains LATIN letters (A–Z), convert them to lowercase (e.g., MDF -> mdf, PVC/ABS -> pvc/abs).\n"
            "   - Keep digits as-is (no Persian digit conversion) and keep any existing Persian words/units inside the value unchanged.\n"
            "4) Never restate or answer the question; return only the normalized VALUE.\n"
            "\n"
            "Examples:\n"
            "- Value: \"MDF\"            -> mdf\n"
            "- Value: MDF               -> mdf\n"
            "- Value: \"PVC/ABS\"        -> pvc/abs\n"
            "- Value: 750 وات           -> 750 وات\n"
            "- Value: \"استیل ضد زنگ\"   -> استیل ضد زنگ\n"
        )

        user = (
            f"User's original query: \"{query}\"\n"
            f"They are asking about the attribute '{attr}'.\n"
            f"The value for this attribute is: '{value}'"
        )
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.1,
        )
        ans = completion.choices[0].message.content if completion.choices else None
        return ans.strip() if ans else None
    except Exception as e:
        logging.info(f"attribute_answer_via_llm skipped: {e}")
        return None


# -------------------- Price detection --------------------

_PRICE_RE = re.compile(
    r'((کمترین|حداقل|پایین\s*ترین|ارزان\s*ترین|متوسط)\s+)?قیمت|چنده|چقدره',
    re.IGNORECASE
)


def is_price_query(query: str) -> bool:

    q = normalize_fa(query or "")
    return bool(_PRICE_RE.search(q))


def _to_number(v) -> Optional[float]:

    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    s = s.replace(',', '').replace('٬', '').replace(' ', '')
    s = s.replace('٫', '.')
    s = _normalize_persian_numerals(s)
    try:
        return float(s)
    except Exception:
        return None


def get_min_price_for_base(base_key: str, members_df: pd.DataFrame) -> Optional[Union[int, float]]:
    if members_df is None or members_df.empty or not base_key:
        return None
    mask = members_df.get('base_random_key', pd.Series(dtype=str)).astype(str) == str(base_key)
    subset = members_df.loc[mask]
    if subset.empty:
        return None

    if 'price' not in subset.columns:
        return None

    prices = subset['price'].map(_to_number).dropna()
    if prices.empty:
        return None
    m = float(prices.min())
    return int(m) if m.is_integer() else m


# -------------------- Warranty count detection --------------------
_WARRANTY_COUNT_HINTS = (
    "چند فروشگاه", "تعداد فروشگاه", "چندتا فروشگاه", "چند تا فروشگاه",
    "فروشگاه با گارانتی", "با گارانتی", "دارای گارانتی"
)

_WARRANTY_TERMS = {
    "گارانتی", "ضمانت", "گارانتی‌دار", "ضمانت‌دار",
    "ضمانت نامه", "ضمانت‌نامه", "گارانتی معتبر"
}


def is_warranty_count_query(query: str) -> bool:

    q = normalize_fa(query or "")
    mentions_warranty = any(t in q for t in _WARRANTY_TERMS)
    return mentions_warranty and any(h in q for h in _WARRANTY_COUNT_HINTS)


def get_warranty_shop_count_for_base(
        base_key: str,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame
) -> Optional[int]:

    try:
        if not base_key or members_df is None or shops_df is None:
            return None

        if members_df.empty or shops_df.empty:
            return None

        base_mask = members_df.get('base_random_key', pd.Series([], dtype=str)).astype(str) == str(base_key)
        subset = members_df.loc[base_mask, ['shop_id']].dropna()
        if subset.empty:
            return 0

        merged = subset.merge(
            shops_df.rename(columns={'id': 'shop_id'}),
            on='shop_id',
            how='left'
        )
        valid = merged.loc[merged['has_warranty'] == True, 'shop_id'].dropna().astype('Int64').unique()
        return int(len(valid))
    except Exception as e:
        logging.exception(f"get_warranty_shop_count_for_base failed: {e}")
        return None


_SHOP_COUNT_HINTS = (
    "چند فروشگاه", "تعداد فروشگاه", "چندتا فروشگاه", "چند تا فروشگاه",
    "چند فروشنده", "تعداد فروشنده", "چند مغازه",
    "چند عضو", "تعداد عضو", "اعضا"
)


def is_shop_count_query(query: str) -> bool:

    q = normalize_fa(query or "")
    if any(t in q for t in _WARRANTY_TERMS):
        return False
    return any(h in q for h in _SHOP_COUNT_HINTS)




def get_shop_count_for_base(
        base_key: str,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame | None = None,
        cities_df: pd.DataFrame | None = None,
        *,
        city: str | None = None,
) -> int | None:
    """
    Count UNIQUE shops listing this base product.
    Works with string shop IDs (recommended) or numeric;
    joins & counts are done on strings to avoid NaNs from numeric coercion.
    """
    try:
        if not base_key or members_df is None or members_df.empty:
            logging.info("Shop-count: base_key=%s -> empty members_df or no key", base_key)
            return 0

        base_mask = members_df.get('base_random_key', pd.Series([], dtype=str)).astype(str) == str(base_key)
        subset = members_df.loc[base_mask, ['shop_id']].copy()
        if subset.empty:
            logging.info("Shop-count detail: base_key=%s city=%s unique_shops=0 filtered_rows=0", base_key, city or "-")
            return 0

        subset['shop_id'] = subset['shop_id'].astype(str).str.strip()
        subset = subset[subset['shop_id'].ne('') & subset['shop_id'].str.lower().ne('nan')]

        if city:
            if shops_df is None or shops_df.empty or not {"id", "city_id"}.issubset(shops_df.columns):
                logging.info("Shop-count: city requested but shops_df invalid/empty")
                return 0

            wanted_city_id = None
            if cities_df is not None and not cities_df.empty and {"id", "name"}.issubset(cities_df.columns):
                cdf = cities_df.copy()
                cdf["__name_norm"] = cdf["name"].astype(str).map(normalize_fa).str.strip()
                row = cdf.loc[cdf["__name_norm"] == normalize_fa(city), ["id"]]
                if not row.empty:
                    wanted_city_id = str(row["id"].iloc[0])
            if wanted_city_id is None:
                logging.info("Shop-count: city '%s' not found in cities_df", city)
                return 0

            shops_small = shops_df.rename(columns={"id": "shop_id"})[["shop_id", "city_id"]].copy()
            shops_small["shop_id"] = shops_small["shop_id"].astype(str).str.strip()
            shops_small["city_id"] = shops_small["city_id"].astype(str).str.strip()

            subset = subset.merge(shops_small, on="shop_id", how="left", validate="many_to_one")
            subset = subset.loc[subset["city_id"].astype(str).str.strip() == wanted_city_id, ["shop_id"]]
            if subset.empty:
                logging.info("Shop-count detail: base_key=%s city=%s unique_shops=0 filtered_rows=0", base_key, city)
                return 0

        unique_shop_count = int(subset['shop_id'].nunique())
        logging.info(
            "Shop-count detail: base_key=%s city=%s unique_shops=%d filtered_rows=%d",
            base_key, city or "-", unique_shop_count, len(subset),
        )
        return unique_shop_count
    except Exception as e:
        logging.exception("get_shop_count_for_base failed (base_key=%s): %s", base_key, e)
        return None


# ---------- Price-type intent & city extraction ----------

_PRICE_TYPE_RULES = [
    ("min", re.compile(r"(حداقل|کمترین|پایین\s*ترین|ارزان\s*ترین)", re.IGNORECASE)),
    ("avg", re.compile(r"(میانگین|متوسط)", re.IGNORECASE)),
    ("max", re.compile(r"(حداکثر|بیشترین|بالا\s*ترین|گران\s*ترین)", re.IGNORECASE)),
]


def detect_price_type(query: str) -> str:
    q = normalize_fa(query or "")
    for label, pat in _PRICE_TYPE_RULES:
        if pat.search(q):
            return label
    return "min"


def extract_city_from_query(query: str, cities_df: pd.DataFrame | None) -> str | None:

    q = normalize_fa(query or "")
    if not q:
        return None

    names: list[str] = []
    if cities_df is not None and not cities_df.empty:
        name_cols = [c for c in ["name", "city", "city_name", "fa_name"] if c in cities_df.columns]
        if name_cols:
            seen = set()
            for c in name_cols:
                vals = cities_df[c].dropna().astype(str).tolist()
                for v in vals:
                    vn = normalize_fa(v)
                    if vn and vn not in seen:
                        seen.add(vn)
                        names.append(vn)

    def _word_boundary_pattern(city: str) -> str:
        inner = r"\s*".join(map(re.escape, city.split()))
        return rf"(?<!\S){inner}(?!\S)"

    if names:
        for city in sorted(names, key=len, reverse=True):
            if re.search(_word_boundary_pattern(city), q):
                return city

    m = re.search(r"(?:در\s+)?(?:شهر\s+)([\u0600-\u06FFA-Za-z]+(?:\s+[\u0600-\u06FFA-Za-z]+){0,2})", q)
    if m:
        candidate = normalize_fa(m.group(1))
        if candidate:
            if names:
                if candidate in names:
                    return candidate
                for city in sorted(names, key=len, reverse=True):
                    if city in candidate or candidate in city:
                        return city
            return candidate

    return None


def _aggregate_prices(subset: pd.DataFrame, *, aggregate: str) -> pd.Series:

    prices = subset.assign(__price=subset["price"].map(_to_number)).dropna(subset=["__price"])
    if prices.empty:
        return pd.Series(dtype=float)

    if aggregate == "per_offer" or "shop_id" not in prices.columns:
        return prices["__price"]

    grp = prices.groupby("shop_id")["__price"]
    if aggregate == "per_shop_mean":
        return grp.mean()
    return grp.min()


def compute_price_stats_for_base(
        base_key: str,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame | None = None,
        *,
        city: str | None = None,
        cities_df: pd.DataFrame | None = None,
        aggregate: str = "per_shop_min",  # 'per_shop_min' | 'per_shop_mean' | 'per_offer'
) -> dict | None:


    def _cast_num(x: float) -> int | float:
        return int(x) if float(x).is_integer() else x

    def _aggregate_prices(subset: pd.DataFrame, *, mode: str) -> pd.Series:

        prices = subset.assign(__price=subset["price"].map(_to_number)).dropna(subset=["__price"])
        if prices.empty:
            return pd.Series(dtype=float)

        if mode == "per_offer" or "shop_id" not in prices.columns:
            return prices["__price"]

        grp = prices.groupby("shop_id")["__price"]
        if mode == "per_shop_mean":
            return grp.mean()
        return grp.min()

    try:
        if not base_key or members_df is None or members_df.empty:
            logging.info("price-stats: missing base_key or empty members_df")
            return None

        base_mask = members_df.get("base_random_key", pd.Series([], dtype=str)).astype(str) == str(base_key)
        subset = members_df.loc[base_mask, ["shop_id", "price"]].copy() if "shop_id" in members_df.columns else \
            members_df.loc[base_mask, ["price"]].copy()
        if subset.empty or "price" not in subset.columns:
            logging.info("price-stats: no offers or 'price' column missing after base filter")
            return None

        if "shop_id" in subset.columns:
            subset["shop_id"] = pd.to_numeric(subset["shop_id"], errors="coerce").astype("Int64")

        if city and shops_df is not None and not shops_df.empty and {"id", "city_id"}.issubset(shops_df.columns):
            wanted_city_id = None

            if cities_df is not None and not cities_df.empty and {"id", "name"}.issubset(cities_df.columns):
                cdf = cities_df.copy()
                cdf["__name_norm"] = cdf["name"].astype(str).map(normalize_fa).str.strip()
                row = cdf.loc[cdf["__name_norm"] == normalize_fa(city), ["id"]]
                if not row.empty:
                    wanted_city_id = row["id"].iloc[0]

            if wanted_city_id is None:
                logging.info("price-stats: city '%s' not found in cities_df; returning empty stats", city)
                return {"count": 0, "min": None, "avg": None, "max": None}

            shops_small = shops_df.rename(columns={"id": "shop_id"})[["shop_id", "city_id"]].copy()
            shops_small["shop_id"] = pd.to_numeric(shops_small["shop_id"], errors="coerce").astype("Int64")

            shops_in_city = shops_small.loc[shops_small["city_id"] == wanted_city_id, ["shop_id"]]
            if shops_in_city.empty:
                return {"count": 0, "min": None, "avg": None, "max": None}

            if "shop_id" in subset.columns:
                subset = subset.merge(shops_in_city, on="shop_id", how="inner", validate="many_to_one")
                if subset.empty:
                    return {"count": 0, "min": None, "avg": None, "max": None}
            else:
                logging.info("price-stats: 'city' requested but members_df lacks 'shop_id'; returning empty stats")
                return {"count": 0, "min": None, "avg": None, "max": None}

        mode = aggregate if aggregate in {"per_shop_min", "per_shop_mean", "per_offer"} else "per_shop_min"
        series = _aggregate_prices(subset, mode=mode)

        try:
            _prices_df = subset.assign(__price=subset["price"].map(_to_number)).dropna(subset=["__price"])
            all_offers = _prices_df["__price"].tolist()
            by_shop = {}
            if "shop_id" in _prices_df.columns:
                by_shop = _prices_df.groupby("shop_id")["__price"].size().to_dict()
            logging.info("price-stats/offers (pre-aggregation %s): %s", mode, sorted(all_offers))
            logging.info("price-stats/offers by shop: %s", by_shop)
            logging.info("price-stats/used values (after %s): %s", mode, sorted(series.tolist()))
        except Exception:
            pass

        if series.empty:
            return {"count": 0, "min": None, "avg": None, "max": None}

        mn, mx, av = float(series.min()), float(series.max()), float(series.mean())

        try:
            shop_count = int(series.shape[0])
            logging.info(
                "price-stats: base=%s city=%s aggregate=%s count=%d min=%s avg=%s max=%s sample=%s",
                base_key, city or "-", mode, shop_count, mn, av, mx,
                dict(series.head(min(5, shop_count)))
            )
        except Exception:
            pass

        return {"count": int(series.shape[0]), "min": _cast_num(mn), "avg": _cast_num(av), "max": _cast_num(mx)}

    except Exception as e:
        logging.exception("compute_price_stats_for_base failed: %s", e)
        return None


# -------------------- Scenario 5: Product Comparison --------------------

_COMMON_CITIES = ["تهران", "مشهد", "اصفهان", "شیراز", "تبریز", "کرج", "قم", "اهواز"]


def _extract_city_by_lookup(query: str, cities_df: pd.DataFrame | None) -> str | None:
    q = normalize_fa(query or "")
    if cities_df is not None and not cities_df.empty:
        name_cols = [c for c in ["name", "city", "city_name", "fa_name"] if c in cities_df.columns]
        if name_cols:
            city_names = set()
            for c in name_cols:
                city_names.update(
                    normalize_fa(x) for x in cities_df[c].dropna().astype(str).tolist()
                )
            for cname in city_names:
                if cname and cname in q:
                    return cname

    for cname in _COMMON_CITIES:
        if cname in q:
            return cname
    return None


def extract_city_and_price_metric_for_comparison(
        query: str,
        cities_df: pd.DataFrame | None
) -> tuple[str | None, str]:

    city = extract_city_from_query(query, cities_df)
    if not city:
        city = _extract_city_by_lookup(query, cities_df)

    metric = detect_price_type(query)
    return city, metric


# ---------- Price aggregation for a specific product in a *specific city ----------

def compute_city_price_for_product(
        product_key: str,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame | None,
        cities_df: pd.DataFrame | None,
        *,
        city: str | None,
        metric: str = "avg",  # 'min' | 'avg' | 'max'
        per_shop_strategy: str = "per_shop_min",
) -> tuple[float | int | None, int]:

    if members_df is None or members_df.empty or not product_key:
        return (None, 0)

    prod_key_col = (
        "base_random_key" if "base_random_key" in members_df.columns else
        ("product_key" if "product_key" in members_df.columns else
         ("random_key" if "random_key" in members_df.columns else None))
    )
    if not prod_key_col or "price" not in members_df.columns:
        return (None, 0)

    sub = members_df.loc[
        members_df[prod_key_col].astype(str) == str(product_key),
        ["shop_id", "price"]
    ].copy() if "shop_id" in members_df.columns else members_df.loc[
        members_df[prod_key_col].astype(str) == str(product_key),
        ["price"]
    ].copy()

    if sub.empty:
        return (None, 0)

    if city:
        if shops_df is not None and not shops_df.empty and {"id", "city_id"}.issubset(shops_df.columns):
            wanted_city_id = None
            if cities_df is not None and not cities_df.empty and {"id", "name"}.issubset(cities_df.columns):
                cdf = cities_df.copy()
                cdf["__name_norm"] = cdf["name"].astype(str).map(normalize_fa).str.strip()
                row = cdf.loc[cdf["__name_norm"] == normalize_fa(city), ["id"]]
                if not row.empty:
                    wanted_city_id = row["id"].iloc[0]

            if wanted_city_id is None:
                return (None, 0)

            if "shop_id" not in sub.columns:
                return (None, 0)

            shops_small = shops_df.rename(columns={"id": "shop_id"})[["shop_id", "city_id"]].copy()
            shops_small["shop_id"] = pd.to_numeric(shops_small["shop_id"], errors="coerce").astype("Int64")
            sub["shop_id"] = pd.to_numeric(sub["shop_id"], errors="coerce").astype("Int64")
            sub = sub.merge(shops_small, on="shop_id", how="left", validate="many_to_one")
            sub = sub.loc[sub["city_id"] == wanted_city_id, ["shop_id", "price"]]
            if sub.empty:
                return (None, 0)
        else:
            return (None, 0)

    sub = sub.assign(__price=sub["price"].map(_to_number)).dropna(subset=["__price"])
    if sub.empty:
        return (None, 0)

    if "shop_id" in sub.columns:
        grp = sub.groupby("shop_id")["__price"]
        series = grp.min() if per_shop_strategy == "per_shop_min" else (
            grp.mean() if per_shop_strategy == "per_shop_mean" else sub["__price"]
        )
    else:
        series = sub["__price"]

    if series.empty:
        return (None, 0)

    if metric == "min":
        val = float(series.min())
    elif metric == "max":
        val = float(series.max())
    else:
        val = float(series.mean())

    val = int(val) if val.is_integer() else val
    return (val, int(series.shape[0]))


COMPARISON_DETECTION_PROMPT = (
    "You are a product comparison detector. Analyze the Persian text to determine if the user is comparing multiple specific products.\n"
    "\n"
    "DETECTION CRITERIA:\n"
    "- User must be comparing 2 or more SPECIFIC products (with names, codes, or clear identifiers)\n"
    "- Must contain comparison words like: کدام، کدوم، بهتر، مناسب‌تر، برتر، ترجیح، انتخاب، مقایسه\n"
    "- Must mention multiple products explicitly (not just asking about one product)\n"
    "- Must be asking which one is better for a specific use case or criteria\n"
    "\n"
    "RESPONSE FORMAT:\n"
    "If this IS a comparison query, respond with: YES\n"
    "If this is NOT a comparison query, respond with: NO\n"
    "\n"
    "Examples:\n"
    "«کدام یک از این ماگ‌های خرید ماگ-لیوان هندوانه فانتزی و کارتونی کد 1375 یا ماگ لته خوری سرامیکی با زیره کد 741 دارای سبک کارتونی و فانتزی بوده و برای کودکان یا نوجوانان مناسب‌تر است؟» → YES\n"
    "«میز تحریر چوبی مدل A100 بهتر است یا میز فلزی مدل B200 برای محیط اداری؟» → YES\n"
    "«ماگ لته خوری چه رنگی دارد؟» → NO\n"
    "«قیمت گوشی سامسونگ چقدر است؟» → NO\n"
)

PRODUCT_EXTRACTION_PROMPT = (
    "Extract all product names from this Persian comparison query. Return them as a JSON list.\n"
    "\n"
    "INSTRUCTIONS:\n"
    "- Extract each distinct product mentioned in the query\n"
    "- Clean up each product name (remove extra words, keep essential identifiers)\n"
    "- Include model numbers, codes, or specific identifiers\n"
    "- Return as JSON array: [\"product1\", \"product2\", ...]\n"
    "\n"
    "Example:\n"
    "Input: «کدام یک از ماگ-لیوان هندوانه فانتزی کد 1375 یا ماگ لته خوری سرامیکی کد 741 بهتر است؟»\n"
    "Output: [\"ماگ-لیوان هندوانه فانتزی کد 1375\", \"ماگ لته خوری سرامیکی کد 741\"]\n"
)

COMPARISON_DECISION_PROMPT = (
    "You are a product comparison expert. Based on the user's question and product information, determine which product is better and explain why.\n"
    "\n"
    "INSTRUCTIONS:\n"
    "1. Carefully analyze the user's specific use case or criteria mentioned in their question\n"
    "2. Compare the products only on features directly relevant to that use case\n"
    "3. Make a clear and decisive choice about which product is better\n"
    "4. Provide a natural, easy-to-understand explanation in Persian that highlights the key advantage of the chosen product\n"
    "\n"
    "RESPONSE FORMAT:\n"
    "Respond with JSON:\n"
    "{\n"
    "  \"winner_key\": \"random_key_of_winning_product\",\n"
    "  \"explanation\": \"Persian explanation of why this product is better\"\n"
    "}\n"
    "\n"
    "EXPLANATION GUIDELINES:\n"
    "- Keep explanation short and clear (1-2 sentences)\n"
    "- Directly connect the product's advantage to the user's mentioned need\n"
    "- Use simple, natural Persian (avoid technical jargon if not needed)\n"
    "- Avoid repeating product names; focus on the benefit or feature instead\n"
    "- Prioritize clarity over detail, but make the reason convincing\n"
)


async def detect_comparison_intent(query: str) -> bool:
    try:
        if not client or not getattr(settings, "openai_model", None):

            q = normalize_fa(query or "")
            comparison_words = ["کدام", "کدوم", "بهتر", "مناسب‌تر", "برتر", "ترجیح", "انتخاب", "مقایسه"]
            has_comparison = any(word in q for word in comparison_words)
            has_multiple_products = q.count("یا") > 0 or q.count("و") > 1
            return has_comparison and has_multiple_products

        completion = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": COMPARISON_DETECTION_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )

        result = completion.choices[0].message.content if completion.choices else None
        return result and result.strip().upper() == "YES"

    except Exception as e:
        logging.info(f"Comparison detection failed, using fallback: {e}")
        q = normalize_fa(query or "")
        comparison_words = ["کدام", "کدوم", "بهتر", "مناسب‌تر", "برتر", "ترجیح", "انتخاب", "مقایسه"]
        has_comparison = any(word in q for word in comparison_words)
        has_multiple_products = q.count("یا") > 0 or q.count("و") > 1
        return has_comparison and has_multiple_products


async def extract_products_from_comparison(query: str) -> List[str]:
    try:
        if not client or not getattr(settings, "openai_model", None):

            text = normalize_fa(query or "")
            raw_parts = re.split(r'\s*(?:یا|،|,)\s*', text)
            products: list[str] = []
            for part in raw_parts:
                p = re.sub(r'[؟?].*$', '', part).strip(' "\'()[]')
                p = re.sub(r'^(?:کدام|کدوم|بهتر|مناسب‌تر|برتر|ترجیح|انتخاب|مقایسه)\s+', '', p).strip()
                if p:
                    products.append(enforce_ascii_digits_in_latin_tokens(p))
            seen = set()
            products = [x for x in products if not (x in seen or seen.add(x))]
            return products[:4]

        completion = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": PRODUCT_EXTRACTION_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )

        result = completion.choices[0].message.content if completion.choices else None
        if result:
            try:
                products = json.loads(result.strip())
                return products if isinstance(products, list) else []
            except json.JSONDecodeError:
                pass

        return []

    except Exception as e:
        logging.info(f"Product extraction failed: {e}")
        return []


async def make_comparison_decision(
        query: str,
        products_info: List[Dict[str, Any]]
) -> Optional[Dict[str, str]]:
    try:
        if not client or not getattr(settings, "openai_model", None) or len(products_info) < 2:
            if products_info:
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return {
                    "winner_key": products_info[0]["random_key"],
                    "explanation": "این محصول ویژگی‌های مناسب‌تری دارد."
                }
            return None

        products_summary = []
        for i, prod in enumerate(products_info, 1):
            features_str = ""
            if prod.get("extra_features"):
                features = parse_extra_features(prod["extra_features"])
                features_str = " - " + ", ".join(f"{k}: {v}" for k, v in features.items() if v)

            products_summary.append(f"محصول {i}: {prod['persian_name']} (کد: {prod['random_key']}){features_str}")

        context = f"سوال کاربر: {query}\n\nمحصولات موجود:\n" + "\n".join(products_summary)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        completion = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": COMPARISON_DECISION_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.1,
        )

        result = completion.choices[0].message.content if completion.choices else None
        if result:
            raw = result.strip()

            try:
                decision = json.loads(raw)
                if isinstance(decision, dict) and "winner_key" in decision and "explanation" in decision:
                    keys = {p["random_key"] for p in products_info}
                    if decision["winner_key"] in keys:
                        return decision
            except json.JSONDecodeError:
                pass

            import re

            keys = [p["random_key"] for p in products_info]
            key_set = set(keys)

            key_match = re.search(r'"winner_key"\s*:\s*"([^"]+)"', raw)
            winner_key = key_match.group(1) if key_match else None
            if winner_key not in key_set:
                winner_key = None

            if not winner_key:
                for k in keys:
                    if k in raw:
                        winner_key = k
                        break

            if not winner_key:
                idx_match = re.search(r"محصول\s+(\d+)", raw)
                if idx_match:
                    idx = int(idx_match.group(1)) - 1
                    if 0 <= idx < len(products_info):
                        winner_key = products_info[idx]["random_key"]

            exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw)
            explanation = exp_match.group(1).strip() if exp_match else None

            if not explanation:
                parts = re.split(r"[.!؟?]\s+", raw)
                explanation = " ".join(parts[:2]).strip()
                if not explanation:
                    explanation = raw[:300].strip()

            if not winner_key:
                winner_key = products_info[0]["random_key"]

            return {
                "winner_key": winner_key,
                "explanation": explanation
            }

        return {
            "winner_key": products_info[0]["random_key"],
            "explanation": "این محصول انتخاب بهتری است."
        }

    except Exception as e:
        logging.info(f"Comparison decision failed: {e}")
        if products_info:
            return {
                "winner_key": products_info[0]["random_key"],
                "explanation": "این محصول انتخاب بهتری است."
            }
        return None


_COVERAGE_COMPARE_HINTS = (
    "شهرهای بیشتری", "تعداد بیشتری از شهرها", "در چند شهر", "تعداد شهر", "بیشترین شهر",
    "در شهرهای", "پوشش شهری", "پوشش شهرها"
)


def detect_city_coverage_intent(query: str, cities_df: pd.DataFrame | None) -> tuple[bool, str | None]:

    q = normalize_fa(query or "")
    if not q:
        return (False, None)

    looks_coverage = ("شهر" in q) and any(h in q for h in _COVERAGE_COMPARE_HINTS)
    if not looks_coverage:
        return (False, None)

    city = extract_city_from_query(q, cities_df)
    return (True, city)


def compute_city_coverage_for_product(
        product_key: str,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame | None,
        cities_df: pd.DataFrame | None,
        *,
        city: str | None = None,
) -> int:

    if not product_key or members_df is None or members_df.empty:
        return 0

    key_col = (
        "base_random_key" if "base_random_key" in members_df.columns else
        ("product_key" if "product_key" in members_df.columns else
         ("random_key" if "random_key" in members_df.columns else None))
    )
    if not key_col or "price" not in members_df.columns:
        pass

    cols = ["shop_id"]
    if "shop_id" not in members_df.columns:
        return 0

    sub = members_df.loc[members_df[key_col].astype(str) == str(product_key), cols].copy()
    if sub.empty:
        return 0

    if shops_df is None or shops_df.empty or "id" not in shops_df.columns or "city_id" not in shops_df.columns:
        return 0

    shops_small = shops_df.rename(columns={"id": "shop_id"})[["shop_id", "city_id"]].copy()
    shops_small["shop_id"] = pd.to_numeric(shops_small["shop_id"], errors="coerce").astype("Int64")
    sub["shop_id"] = pd.to_numeric(sub["shop_id"], errors="coerce").astype("Int64")
    sub = sub.merge(shops_small, on="shop_id", how="left", validate="many_to_one")
    if sub.empty:
        return 0

    if city:
        wanted_city_id = None
        if cities_df is not None and not cities_df.empty and {"id", "name"}.issubset(cities_df.columns):
            cdf = cities_df.copy()
            cdf["__name_norm"] = cdf["name"].astype(str).map(normalize_fa).str.strip()
            row = cdf.loc[cdf["__name_norm"] == normalize_fa(city), ["id"]]
            if not row.empty:
                wanted_city_id = row["id"].iloc[0]
        if wanted_city_id is None:
            return 0

        in_city = sub.loc[sub["city_id"] == wanted_city_id, ["shop_id"]]
        return int(in_city["shop_id"].dropna().astype("Int64").nunique())

    return int(sub["city_id"].dropna().astype("Int64").nunique())


_EXPLICIT_KEY_RE = re.compile(r'(?:شناسه|random\s*key|base\s*key)\s*[\"“”]?\s*([A-Za-z0-9_-]{4,})\s*[\"“”]?',
                              re.IGNORECASE)


def extract_explicit_random_keys(text: str) -> list[str]:
    q = normalize_fa(text or "")
    return list(dict.fromkeys(m.group(1) for m in _EXPLICIT_KEY_RE.finditer(q)))


def get_shop_count_for_any_key(
        key: str,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame | None = None,
        cities_df: pd.DataFrame | None = None,
        *,
        city: str | None = None,
) -> int:

    if not key or members_df is None or members_df.empty:
        return 0

    key_col = None
    for col in ("base_random_key", "product_key", "random_key"):
        if col in members_df.columns:
            sub = members_df.loc[members_df[col].astype(str) == str(key), ["shop_id"]].copy()
            if not sub.empty:
                key_col = col
                subset = sub
                break

    if key_col is None:
        return 0

    if "shop_id" not in subset.columns:
        return 0

    subset["shop_id"] = pd.to_numeric(subset["shop_id"], errors="coerce").astype("Int64")

    if city:
        if shops_df is None or shops_df.empty or not {"id", "city_id"}.issubset(shops_df.columns):
            return 0
        wanted_city_id = None
        if cities_df is not None and not cities_df.empty and {"id", "name"}.issubset(cities_df.columns):
            cdf = cities_df.copy()
            cdf["__name_norm"] = cdf["name"].astype(str).map(normalize_fa).str.strip()
            row = cdf.loc[cdf["__name_norm"] == normalize_fa(city), ["id"]]
            if not row.empty:
                wanted_city_id = row["id"].iloc[0]
        if wanted_city_id is None:
            return 0

        shops_small = shops_df.rename(columns={"id": "shop_id"})[["shop_id", "city_id"]].copy()
        shops_small["shop_id"] = pd.to_numeric(shops_small["shop_id"], errors="coerce").astype("Int64")
        subset = subset.merge(shops_small, on="shop_id", how="left", validate="many_to_one")
        subset = subset.loc[subset["city_id"] == wanted_city_id, ["shop_id"]]
        if subset.empty:
            return 0

    return int(subset["shop_id"].dropna().astype("Int64").nunique())


async def handle_product_comparison(
        query: str,
        products_df: pd.DataFrame,
        members_df: pd.DataFrame,
        shops_df: pd.DataFrame,
        cities_df: pd.DataFrame
) -> Optional[ChatResponse]:
    try:
        if not await detect_comparison_intent(query):
            return None
        logging.info("Detected comparison intent")

        product_names = await extract_products_from_comparison(query)
        logging.info(f"Extracted products (names): {product_names}")

        explicit_keys = extract_explicit_random_keys(query)
        logging.info(f"Explicit random keys in text: {explicit_keys}")

        found_products: list[Dict[str, Any]] = []

        def _row_base_key(row: pd.Series) -> str:
            base_key = ""
            if "base_random_key" in products_df.columns:
                base_key = str(row.get("base_random_key", "")).strip()
            if not base_key:
                base_key = str(row.get("random_key", "")).strip()
            return base_key

        for product_name in product_names[:4]:
            keys = find_product_key_in_df(product_name, products_df)
            if keys:
                row = products_df[products_df['random_key'] == keys[0]].iloc[0]
                found_products.append({
                    "random_key": str(keys[0]),
                    "persian_name": row.get('persian_name', ''),
                    "extra_features": row.get('extra_features', {}),
                    "base_random_key": _row_base_key(row),
                })
                logging.info(f"found_products in text: {found_products}")

        for rk in explicit_keys:
            sub = products_df[products_df['random_key'].astype(str) == str(rk)]
            if not sub.empty:
                row = sub.iloc[0]
                if not any(p["random_key"] == rk for p in found_products):
                    found_products.append({
                        "random_key": str(rk),
                        "persian_name": row.get('persian_name', ''),
                        "extra_features": row.get('extra_features', {}),
                        "base_random_key": _row_base_key(row),
                    })

        if len(found_products) < 2:
            logging.info(f"Found only {len(found_products)} products for comparison")
            return None

        is_cov, city = detect_city_coverage_intent(query, cities_df)
        if is_cov:
            stats: list[tuple[str, int]] = []
            for p in found_products:
                key_for_members = p.get("base_random_key") or p["random_key"]
                metric = compute_city_coverage_for_product(
                    key_for_members, members_df, shops_df, cities_df, city=city
                )
                stats.append((p["random_key"], int(metric)))

            stats_sorted = sorted(stats, key=lambda x: x[1], reverse=True)
            winner_key, winner_val = stats_sorted[0]
            if city:
                explanation = f"در {city}، این گزینه در فروشگاه‌های بیشتری موجود است ({winner_val})."
            else:
                explanation = f"این گزینه پوشش شهری بیشتری دارد ({winner_val} شهر)."

            logging.info(
                f"Comparison decision (coverage-based): winner={winner_key}, metric={winner_val}, city={city or '-'}")
            return ChatResponse(message=explanation, base_random_keys=[winner_key])

        q_norm = normalize_fa(query or "")
        wants_shop_count = (
                is_shop_count_query(q_norm)
                or any(h in q_norm for h in _SHOP_COUNT_HINTS)
                or ("به لحاظ" in q_norm and (
                    "تعداد فروشگاه" in q_norm or "فروشگاه‌ها" in q_norm or "فروشگاهها" in q_norm))
        )

        if wants_shop_count:
            city_sc = extract_city_from_query(query, cities_df)

            def _count_shops_for_product(p: Dict[str, Any]) -> int:

                base_key = (p.get("base_random_key") or "").strip()
                if base_key:
                    cnt = get_shop_count_for_base(base_key, members_df, shops_df, cities_df, city=city_sc)
                    return int(cnt or 0)

                key = (p.get("random_key") or "").strip()
                if not key or members_df is None or members_df.empty:
                    return 0
                sub = None
                for col in ("product_key", "random_key", "base_random_key"):
                    if col in members_df.columns:
                        sub = members_df.loc[members_df[col].astype(str) == key, ["shop_id"]].copy()
                        if not sub.empty:
                            break
                if sub is None or sub.empty or "shop_id" not in sub.columns:
                    return 0

                sub["shop_id"] = pd.to_numeric(sub["shop_id"], errors="coerce").astype("Int64")

                if city_sc:
                    if shops_df is None or shops_df.empty or not {"id", "city_id"}.issubset(shops_df.columns):
                        return 0
                    wanted_city_id = None
                    if cities_df is not None and not cities_df.empty and {"id", "name"}.issubset(cities_df.columns):
                        cdf = cities_df.copy()
                        cdf["__name_norm"] = cdf["name"].astype(str).map(normalize_fa).str.strip()
                        row = cdf.loc[cdf["__name_norm"] == normalize_fa(city_sc), ["id"]]
                        if not row.empty:
                            wanted_city_id = row["id"].iloc[0]
                    if wanted_city_id is None:
                        return 0
                    shops_small = shops_df.rename(columns={"id": "shop_id"})[["shop_id", "city_id"]].copy()
                    shops_small["shop_id"] = pd.to_numeric(shops_small["shop_id"], errors="coerce").astype("Int64")
                    sub = sub.merge(shops_small, on="shop_id", how="left", validate="many_to_one")
                    sub = sub.loc[sub["city_id"] == wanted_city_id, ["shop_id"]]
                    if sub.empty:
                        return 0

                return int(sub["shop_id"].dropna().astype("Int64").nunique())

            results: list[tuple[str, int]] = []
            for p in found_products:
                cnt = _count_shops_for_product(p)
                results.append((p["random_key"], cnt))
                logging.info("Shop-count compare detail: key=%s shops=%d", p.get("base_random_key") or p["random_key"],
                             cnt)

            winner_key, winner_cnt = max(results, key=lambda x: x[1]) if results else (
            found_products[0]["random_key"], 0)

            logging.info(
                "Scenario 5 (shop-count compare): city=%s results=%s winner=%s cnt=%d",
                city_sc or "ALL", results, winner_key, winner_cnt
            )

            city_part = f" در {city_sc}" if city_sc else ""
            explanation = f"به لحاظ تعداد فروشگاه‌ها{city_part}، این گزینه بهتر است ({to_persian_digits(winner_cnt)} فروشگاه)."
            return ChatResponse(message=explanation, base_random_keys=[winner_key])

        looks_pricey = ("قیمت" in q_norm) or ("چنده" in q_norm) or ("چقدره" in q_norm) or any(
            pat.search(q_norm) for _, pat in _PRICE_TYPE_RULES
        )
        city_price, metric = extract_city_and_price_metric_for_comparison(query, cities_df)

        if looks_pricey:
            stats: list[tuple[str, float | int | None, int]] = []
            for p in found_products:
                prod_key_for_members = p.get("base_random_key") or p["random_key"]
                val, cnt = compute_city_price_for_product(
                    prod_key_for_members, members_df, shops_df, cities_df,
                    city=city_price, metric=metric, per_shop_strategy="per_shop_min",
                )
                logging.info("Price-compare detail: key=%s shops_used=%d value=%s", prod_key_for_members, cnt, val)
                stats.append((p["random_key"], val, cnt))

            usable = [(k, v, c) for (k, v, c) in stats if v is not None and c > 0]
            if usable:
                if metric in ("min", "avg"):
                    winner_key, winner_val, _ = sorted(usable, key=lambda x: float(x[1]))[0]
                    better_word = "کم‌تر";
                    metric_fa = "کمینه" if metric == "min" else "میانگین"
                else:
                    winner_key, winner_val, _ = sorted(usable, key=lambda x: float(x[1]), reverse=True)[0]
                    better_word = "بیش‌تر";
                    metric_fa = "بیشینه"
                city_part = f" در {city_price}" if city_price else ""
                if isinstance(winner_val, (int, float)) and float(winner_val).is_integer():
                    val_str = f"{int(winner_val):,}"
                else:
                    try:
                        val_str = f"{winner_val:,}"
                    except Exception:
                        val_str = str(winner_val)
                explanation = f"با توجه به {metric_fa} قیمت{city_part}، این گزینه {better_word} است ({val_str})."
                logging.info(
                    f"Comparison decision (price-based): winner={winner_key}, value={winner_val}, city={city_price}, metric={metric}")
                return ChatResponse(message=explanation, base_random_keys=[winner_key])

        decision = await make_comparison_decision(query, found_products)
        if not decision:
            return None
        return ChatResponse(message=decision["explanation"], base_random_keys=[decision["winner_key"]])

    except Exception as e:
        logging.exception(f"handle_product_comparison failed: {e}")
        return None


# -------------------- Scenario 6: Main object in an image --------------------

_MAIN_OBJECT_HINTS = [
    "شیء اصلی", "شی اصلی", "موضوع اصلی", "مفهوم اصلی",
    "در تصویر چیست", "تو عکس چیه", "چی در تصویر", "چیه تو تصویر",
    "شیء و مفهوم اصلی", "سوژه اصلی", "عکس"
]


def is_main_object_query(query: str) -> bool:
    q = normalize_fa(query or "")
    return any(h in q for h in _MAIN_OBJECT_HINTS)


import base64, re, logging
from typing import Optional, Dict, Any, Tuple


def _normalize_image_content(content: str) -> Optional[Tuple[str, str]]:

    if not content:
        return None

    content = content.strip()
    mime = "image/jpeg"
    b64 = content

    if content.startswith("data:"):
        try:
            header, b64 = content.split(",", 1)
        except ValueError:
            logging.warning("Invalid data URL (no comma).")
            return None
        m = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64$", header, flags=re.IGNORECASE)
        if not m:
            logging.warning("Unsupported or malformed data URL header: %s", header)
            return None
        mime = m.group(1)

    b64 = re.sub(r"\s+", "", b64)

    try:
        base64.b64decode(b64, validate=True)
    except Exception:
        logging.warning("Invalid image base64 payload.")
        return None

    return mime, b64


async def classify_main_object_from_image_query(query: Dict[str, Any]) -> Optional[Dict[str, str]]:

    try:
        model = getattr(settings, "openai_model", None)
        if client is None or not model:
            return None
        system = (
            "You are a retail image classifier for Persian product categories.\n"
            "Return STRICT JSON with exactly two keys:\n"
            "{ \"coarse_label\": \"<one from list>\", \"fine_label\": \"<short precise persian name>\" }\n\n"

            "DEFINITIONS:\n"
            "- coarse_label: Choose EXACTLY ONE title VERBATIM from the provided CANDIDATE TITLES (Persian). "
            "  If none fits under the rules below, set \"unknown\".\n"
            "- fine_label: A more precise FREEFORM Persian name (1–3 words) for the same object (a subtype / common market name). "
            "  Use a generic Persian noun within the chosen category if unsure (e.g., «لیوان», «کاسه», «پتو»). "
            "  Only set \"unknown\" if the main object itself is unknown.\n\n"

            "TEXT_HINTS: OCR keywords/brands from the image (if provided).\n\n"
            "- Use environmental and companion clues to make a more accurate guess (for fine_label) — but only when they are meaningfully related to the MAIN object (inside the same box, on the same base/plate, connected by a cable/hose, or clearly for the same use).\n"

            "FORCED-CHOICE RULE (VERY IMPORTANT):\n"
            "Pick a category even if confidence is low. Only output coarse_label=\"unknown\" when one of these is true:\n"
            "  A) The image has no clear purchasable object (empty, abstract texture, UI screenshot only),\n"
            "  B) The visible object is out-of-domain for ALL candidates (e.g., animal, car, human portrait),\n"
            "  C) The image is unreadable/corrupted.\n"
            "In all other cases, select the single closest candidate.\n\n"

            "MAIN-OBJECT POLICY (STRICT HIERARCHY):\n"
            "- In multi-object scenes, classify the PURCHASABLE CONTAINER/SUPPORT item if:\n"
            "  1) It occupies >60% of the image area AND shows its full structure (legs/frame/doors/shelves)\n"
            "  2) The supported item is clearly a PROP or ACCESSORY to showcase the main furniture\n"
            "  3) The furniture piece is the obvious selling focus (centered, well-lit, complete)\n"
            "- Classify the SUPPORTED ITEM only if:\n"
            "  1) It's in close-up (>70% of frame) OR\n"
            "  2) The support structure is barely visible/cropped OR\n"
            "  3) There are clear product marketing cues (model numbers, brand labels, product shots)\n"
            "- FURNITURE PRIORITY: میز تلویزیون > کمد و قفسه > صندلی و نیمکت > تلویزیون\n"
            "- When in doubt between furniture and electronics: ask \"What is being sold here - the stand or the device?\"\n"
            "- Priority ladder when multiple apply: Furniture/Storage > Appliances/Devices > Small decor/utensils.\n"

            "VISUAL CONTEXT RULES:\n"
            "- SHOWROOM/CATALOG shots: If the item is clearly staged for sale (clean background, professional lighting, centered composition), classify the primary marketed product\n"
            "- IN-USE scenes: If showing actual usage context (messy kitchen, lived-in room), classify the most prominent functional item\n"
            "- CLOSE-UP product shots: Ignore barely visible supports/containers in frame edges\n"
            "- WIDE establishment shots: Consider what a customer would purchase as the complete item\n"

            "SPECIFIC OVERRIDES:\n"
            "- TV on stand in wide shot with stand fully visible → \"میز تلویزیون\"\n"
            "- TV close-up with only corner of stand → \"تلویزیون\"\n"
            "- Kettle+teapot stacked set → \"کتری و قوری\" (never samovar)\n"
            "- Traditional urn with built-in burner+spigot → \"سماور گازی و زغالی\"\n"
            "- Framed or plaque-style wall art (including music/Spotify-style poster with track name, QR code, player controls, or album art) → "
            "If the candidate list contains \"تابلو و مجسمه / تابلو عکس\", choose it; else prefer \"قاب و شاسی عکس\"; if absent, choose \"تابلو و مجسمه\". "
            "Fine label examples: «تابلو عکس», «پوستر موسیقی», «قاب عکس».\n\n"

            "SELECTION RULES:\n"
            "1) Identify the single main purchasable object (largest / most central / in-focus). "
            "   Ignore background, people, logos, UI icons, QR codes, and decorative text unless the frame/print itself is the product.\n"
            "2) Prefer the most specific candidate over generic ones when multiple could apply. "
            "   Example: if both «ظروف سرو و پذیرایی» and «کاسه و پیاله» exist and the item is bowl-like, choose «کاسه و پیاله».\n"
            "3) If the product is presented INSIDE a gift/presentation box (ribbon, velvet/foam bed, curated set), "
            "   prefer gift/box categories. If «ظرف و باکس هدیه زعفران» exists and saffron cues are present, choose that; "
            "   otherwise choose «ظرف و باکس هدیه» (if present). For fine_label, use a short Persian subtype like «باکس هدیه» یا «ست زعفران».\n"
            "4) For framed prints/posters/plaques: choose «قاب و شاسی عکس». If absent, choose «تابلو و مجسمه». "
            "   fine_label can be a concise subtype like «قاب عکس».\n"
            "5) When multiple items appear, classify ONLY the primary item; never combine labels.\n"
            "6) Do NOT invent labels, translate, or alter spacing/letters. "
            "   coarse_label MUST match a candidate title exactly; fine_label must be Persian (no Latin) and without punctuation.\n"
            "7) Artifact/collectible heuristic (general): if the main object is an artifact whose value is primarily historic/numismatic/philatelic/commemorative "
            "(e.g., small metal disc or medal with relief, date/denomination/country seal; paper with serials/watermarks; engraved badges), "
            "map to the closest collectible category in the list (e.g., «اشیاء قدیمی و کلکسیونی») and set fine_label to the common Persian noun for the object\n"
            "8) For framed prints/posters/plaques: FIRST try \"تابلو و مجسمه / تابلو عکس\" if present. "
            "If absent, choose «قاب و شاسی عکس». If that is also absent, choose «تابلو و مجسمه». "
            "fine_label can be a concise subtype like «تابلو عکس», «پوستر», «قاب عکس».\n"

            "SAMOVAR CLASSIFICATION (VERY RESTRICTIVE):\n"
            "Use سماور categories ONLY when ALL THREE are clearly visible:\n"
            "1) INTEGRATED HEATER: Built-in electric base with cord/controls OR visible firebox/chimney/coal chamber\n"
            "2) TANK GEOMETRY: Large cylindrical reservoir (not just a kettle) with substantial water capacity\n"
            "3) FIXED SPIGOT: Permanently attached tap/faucet (not removable pouring spout)\n"
            "EXPLICIT DISQUALIFIERS for samovar:\n"
            "- If it's a teapot sitting ON TOP of a kettle/warmer → always \"کتری و قوری\"\n"
            "- If heating is external (gas burner underneath, separate warmer) → always \"کتری و قوری\"\n"
            "- If no visible heater mechanism (just spigot + tank) → always \"کتری و قوری\"\n"
            "- If marketed as \"کتری\" or \"قوری\" in visible text → never samovar\n"
            "- Two-piece stackable sets → always \"کتری و قوری\"\n"
            "DEFAULT RULE: When tea equipment is ambiguous, always choose \"کتری و قوری\" unless the 3-signal samovar rule is satisfied with 100% certainty.\n"

            "DISAMBIGUATION — TEA PREP CATEGORIES:\n"
            "- Choose «سماور برقی» یا «سماور گازی و زغالی» فقط وقتی که محصول یک سماور/URN با مخزن بزرگ، گرمکن داخلی (برقی یا شعله)، و شیر خروجی برای ریختن باشد و به‌وضوح به‌عنوان «سماور» بازاریابی شده باشد.\n"
            "- اگر محصول یک ست دو تکه کتری + قوری است که قوری روی کتری/وارمر قرار می‌گیرد و گرمایش اصلی خارجی است (روگازی/وارمر جدا)، دسته‌بندی «کتری و قوری» را انتخاب کن. نمونه fine_label: «ست کتری و قوری»، «کتری روگازی».\n"
            "- وقتی هر دو تفسیر محتمل‌اند (سماور در برابر کتری+قوری) به طور پیش‌فرض «کتری و قوری» را انتخاب کن مگر این‌که شواهد قوی از گرمکن یکپارچه سماور وجود داشته باشد (مثلاً پایه برقی/محفظه مشعل/برندینگ «سماور» روی بدنه).\n"

            "FINAL CONFIDENCE CHECK:\n"
            "Before outputting, ask yourself:\n"
            "1) \"What would a customer primarily be buying here?\"\n"
            "2) \"Is this classification based on the dominant visual element?\"\n"
            "3) \"Am I applying the most restrictive interpretation for ambiguous cases?\"\n"
            "4) \"Does this match the marketing intent of the image?\"\n"
            "If any answer suggests your classification might be wrong, choose the more conservative/general category.\n"

            "OUTPUT SANITY CHECK (must run before you answer):\n"
            "- Ensure coarse_label is EXACTLY one of the provided CANDIDATE TITLES. "
            "  If your tentative choice is not an exact match (spacing/letters differ), correct it to the nearest exact title from the list.\n"
            "- Never invent labels or translate. coarse_label must be an exact candidate string in Persian.\n"
            "- fine_label must be Persian (no Latin) and 1–3 words without punctuation. Use a generic noun within the chosen category if uncertain.\n"

            "OUTPUT FORMAT:\n"
            "- Output JSON ONLY. No prose, no code blocks, no extra keys, no trailing punctuation.\n"
            "- CRITICAL EXAMPLES:\n"
            "- TV mounted on/in entertainment center with shelves/cabinet visible → {\"coarse_label\":\"میز تلویزیون\",\"fine_label\":\"میز تلویزیون\"}\n"
            "- TV close-up shot, stand barely visible → {\"coarse_label\":\"تلویزیون\",\"fine_label\":\"تلویزیون ال ای دی\"}\n"
            "- Stacked kettle+teapot set, no built-in heater → {\"coarse_label\":\"کتری و قوری\",\"fine_label\":\"ست کتری و قوری\"}\n"
            "- Traditional urn with internal fire chamber + spigot → {\"coarse_label\":\"سماور گازی و زغالی\",\"fine_label\":\"سماور سنتی\"}\n"
            "- Wide shot: TV on an open-frame wood+metal unit with shelves and décor visible → {\"coarse_label\":\"میز تلویزیون\",\"fine_label\":\"میز تلویزیون فلزی و چوبی\"}\n"
            "- Close-up: Samsung TV with model text, only a sliver of stand visible → {\"coarse_label\":\"تلویزیون\",\"fine_label\":\"تلویزیون ال ای دی\"}\n"
            "- Small countertop device with a clear dome lid + trays with egg-shaped holes next to raw eggs → {\"coarse_label\":\"بخار پز، آرام پز و هوا‌پز برقی\",\"fine_label\":\"تخم مرغ اب پز کن\"}\n"
            "- Examples of other valid outputs:\n"
            "- Framed music/Spotify-style print with QR and player icons → {\"coarse_label\":\"تابلو و مجسمه / تابلو عکس\",\"fine_label\":\"پوستر موسیقی\"}\n"
            "  {\"coarse_label\":\"ماگ\",\"fine_label\":\"ماگ سرامیکی\"}\n"
            "  {\"coarse_label\":\"کاسه و پیاله\",\"fine_label\":\"کاسه سرامیکی\"}\n"
            "  {\"coarse_label\":\"شمع و جاشمعی\",\"fine_label\":\"شمعدان برنزی\"}\n"
            "  {\"coarse_label\":\"ظرف و باکس هدیه\",\"fine_label\":\"باکس هدیه\"}\n"
            "You will receive an image and a text block named CANDIDATE TITLES containing the allowed labels. "
            "Pick one for coarse_label and provide a concise Persian fine_label if possible."
        )

        incoming = (query or {}).get("messages", [])
        user_content = []
        for m in incoming:
            mtype = (m.get("type") or "").lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if mtype == "text":
                user_content.append({"type": "text", "text": content})
            elif mtype == "image":
                norm = _normalize_image_content(content)
                if not norm:
                    continue
                mime, b64 = norm
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })

        if not user_content:
            logging.warning("No usable user content (text/image) found in query.")
            return None

        ALL_TITLES = "یخچال , کولر گازی , فرش ماشینی , تشک، لحاف و روتختی , قهوه و اسپرسو ساز , مبل استیل و سلطنتی , قالب شیرینی پزی و دسر , بخاری برقی , بخاری , رادیاتور پنلی و پره‌ای , کولر , حوله حمام , روتختی، لحاف و سرویس خواب , تشک تک نفره , پتو تک نفره , کیسه فریزر , نخ سردوز , دسته گل , یخچال فریزر دوقلو , کولر آبی , پتو، بالش و روبالشی , قهوه ساز 1 , داکت اسپلیت , فرش دستبافت , بخاری گازی , هیتر گرمایشی , شوفاژ برقی , حوله دستی , مبل راحتی , تشک دو نفره , پتو دو نفره , کیسه زباله , نخ قیطان , وردنه آشپزی , جام گل , تشک , دریپر قهوه , یخچال فریزر بالاپایین , تصفیه‌‌ هوا , مبل کلاسیک , رادیاتور حوله‌ای , گلیم، جاجیم و گبه , ظروف یکبار مصرف , آتشدان , بخاری نفتی، گازوئیلی و هیزمی , حوله تن پوش , کاور رختخواب , پتو مسافرتی , نخ مکرومه , کپسول کاغذی , کولر و پنکه USB و شارژی , باکس و سبد گل و میوه , اسپرسوساز همراه , یخچال فریزر ساید بای ساید , پنکه , فرنچ پرس , مبل تخت‌خواب شو , فرشینه ، پادری و کناره , چشم بند خواب , هیتر سونا , رادیاتور شیشه‌ای و دکوراتیو , کلاه حمام , سایر انواع پتو , سلفون بسته‌بندی , نخ قرقره , استامپ و کاتر شیرینی , گل رز جاودان , کرسی برقی , فریزر , دستگاه بخور و رطوبت ساز , موکت , موکاپات و قهوه جوش , کاور شوفاژ , شومینه , تشک نوجوان , فویل آلومینیوم , نخ کوبلن و گلدوزی , قیف و ماسوره قنادی , گل مناسبتی , کلاه خواب , یخچال فریزر در فرانسوی , تاج گل , فرش کودک , فن کویل , قطعات و لوازم نصب رادیاتور , مبلمان باغی و فضای باز , حوله آشپزخانه , زیپ کیپ , نخ عمامه , پایه و صفحه گردان شیرینی پزی , قهوه ساز کمکس , استند و قفسه هیزم , هواکش و فن تهویه , فرش آشپزخانه , پاف و بین بگ , لوازم جانبی هیتر و بخاری , سرویس حوله عروس و داماد , نایلون حباب دار , نخ ابریشم , پیمانه آشپزی , قهوه ترک ساز , شاخه گل طبیعی , کف ساز و شیرجوش , کلاژ و فرش پوست , مبلمان کودک , چیلر و مینی چیلر , لوازم جانبی شومینه , دمپایی حوله ای , نایلون دسته دار , دوک خیاطی , کاغذ شیرینی , دستکش یکبار مصرف , روفرشی , مولتی اسپلیت (VRF) , حوله استخری و مسافرتی , مبلمان بادی , نخ ماهیگیری , پالت و کاردک قنادی , قهوه ساز سایفون , فرش نمدی , سایر لوازم قنادی , هواساز , سفره یکبار مصرف , نخ نامرئی , ست حوله و هدبند , قهوه ساز سرد دم , قطعات قهوه ساز , سایر حوله , لوازم جانبی و مصرفی سرو قهوه , سایر مبلمان , فرش قدیمی , رطوبت گیر , پیشبند یکبار مصرف , نخ تریشه , نخ تریکو بافی , حصیر و زیرانداز , دستگاه خوشبو کننده هوا , نخ بافت , فیلتر تصفیه هوا , حوله یکبار مصرف , نخ دوخت , فرش تافتینگ , برج خنک کننده , کلاه یکبار مصرف , فرش چاپی , سایر نخ , ملزومات فرش , قطعات فن کویل , کاور آستین یکبار مصرف , قطعات و لوازم جانبی کولر , جعبه و پاکت غذا , دبه و گالن , قطعات و لوازم جانبی دستگاه بخار و مه ساز , قوطی و بطری پلاستیکی , نی نوشیدنی , قطعات و لوازم جانبی پنکه , جعبه و باکس پلاستیکی , ملحفه و روتختی یکبار مصرف , وان پلاستیکی , سینی یکبار مصرف , پاکت و کاغذ کرافت , کیسه اسپان , نایلون نهال , جا لیوانی یکبار مصرف , هیتر، بخاری و شومینه , شوفاژ و رادیاتور , وسایل پلاستیکی و یکبار مصرف , مبل راحتی چستر , آبسرد کن , وان و جکوزی حمام , بانکه و ظروف بنشن , سینی پذیرایی , صندلی مدیریتی , آبگرمکن گازی , تشک طبی و فنری , پتو , کاموا , سوزن چرخ خیاطی , آبگرمکن , منسوجات آشپزخانه , چای ساز , سماور برقی , مبل راحتی ال , دستگاه تصفیه آب , جای ادویه و پاسماوری , آجیل و شکلات خوری , صندلی کارمندی , آبگرمکن برقی , تشک سنتی و میهمان , بالش , زیر دوشی حمام , قلاب و میل بافتنی , سوزن لحاف دوزی , کتری برقی , سایر مبل راحتی , کابین دوش و سونا , هوزینگ تصفیه آب , جای سیب زمینی و پیاز , شیرینی و میوه خوری , آبگرمکن خورشیدی , صندلی کنفرانس , اجاق و لوازم پخت و پز , روبالشی , تشک نمدی , سوزن گلدوزی , لوازم برقی پخت و پز , کتری و قوری , فیلتر تصفیه آب , قفسه حمام , جای برنج , تخته و ظرف سرو , پکیج دیواری و زمینی , آبگرمکن نفتی , تشک بیمارستانی , سوزن ته مرواریدی , ظروف پخت و پز , سماور گازی و زغالی , لوازم جانبی آبسرد کن , لیف و کیسه حمام , جای نان , مشعل , صندلی آموزشی , محافظ و پد تشک , سرویس پذیرایی , جاسوزنی , آبگرمکن صنعتی , ظروف سرو و پذیرایی , لوازم جانبی دستگاه تصفیه آب , پرده حمام , بطری نوشیدنی , پرده هوا , سایر صندلی اداری , قطعات و لوازم جانبی آبگرمکن , سنجاق قفلی , نوشیدنی‌ ساز , جای حوله و رخت آویز حمام , سرویس آشپزخانه , مبدل حرارتی , سایر انواع تشک , سوزن نخ کن , صندلی و چهار پایه حمام , جت هیتر , ابزار آشپزخانه , ظرف و باکس هدیه زعفران , سوزن دست دوز , تجهیزات جانبی سونا و جکوزی , جای تخم مرغ , دیگ و بویلر , وسایل کاربردی آشپزخانه , سوزن پلاستیکی , لوازم قنادی , پادری و کفپوش حمام , منبع آب گرم , روغن ریز , لوازم یدکی آشپزخانه , یونیت هیتر , برس شستشوی سر و بدن , ظرف نگهداری زیتون و ترشی , وان و جکوزی بادی , تعمیرات لوازم خانگی , کوره هوای گرم , تلمبه چاه باز کن , ظرف نگهداری پاکت شیر , سایر لوازم آشپزخانه و پخت و پز , سایر لوازم حمام , قطعات و لوازم نصب پکیج , اجاق گاز , تلویزیون , پمپ استخری , فرش , میز مدیریت , رگال سقفی , دینام و پدال چرخ خیاطی , کابل و آنتن تلویزیون , لوستر و چراغ تزئینی , سختی‌گیر و رسوب زدا , میز کارمندی , فر برقی و گازی , جای کمربند و کروات , پایه چرخ خیاطی , پودر و ورق ژلاتین , ظروف تزئینی , وانیل , کلر زن استخر , میز کنفرانس , آویز نگهدارنده کیف و کفش , لامپ چرخ خیاطی , گیرنده دیجیتال , گلدان و گل مصنوعی , رنگ خوراکی , ازن زن استخر , ست اجاق گاز و هود , سایر میز اداری , جای البسه و کفش ریلی , ماسوره و ماکو , سایر لوازم جانبی تلویزیون , کاغذ دیواری , پودر کاکائو , تنور , دستگاه UV استخر , روغن و روغن دان چرخ خیاطی , قطعات ماشین آلات دوخت , شمع و جاشمعی , اسانس خوراکی , فیلتر شنی , اجاق برقی , آینه دکوراتیو , بیکینگ پودر , گرمکن استخر و جکوزی , پیک نیک و چراغ خوراک پزی , تابلو و مجسمه , جوش شیرین , چراغ استخر و سونا , کشوی گرمکن , تابلو فرش , گاتر و گریل آب , پودر نارگیل , رومیزی و پرده , آب مرکبات‌گیری , خامه قنادی , اسکیمر استخر , اشیاء قدیمی و کلکسیونی , ساعت دیواری٬ رومیزی و تزئینی , استخر پیش ساخته , نظافت استخر , گل و گیاه طبیعی , دستگیره و نردبان استخر , سایر تجهیزات استخر و آبنما , استند و پایه گلدان , لوازم تزئینی منزل , کاپ و تندیس , جای ریموت کنترل , قاب و شاسی عکس , طلق رومیزی , مبلمان اداری , پلوپز و مولتی کوکر , توالت فرنگی , یخچال و فریزر , سگگ لباس , لوازم آشپزخانه , آب سردکن و تصفیه آب , چرخ گوشت , توالت ایرانی , صندلی اداری , اپلیکه و تکه دوزی , دکوراسیون منزل , تلویزیون و لوازم جانبی , لوازم جانبی یخچال و فریزر , غذا ساز , ست سرویس بهداشتی , میز اداری , مهره و سنگ تزئینی لباس , یخچال، فریزر و آب سردکن , سرخ کن , آینه و روشویی , تریبون , منگوله لباس , شستشو و نظافت , همزن برقی , جای صابون و مایع دستشویی , فایل و کشو اداری , پولک و منجوق , اتاق خواب , مایکروویو , ست جهیزیه , دست خشک کن , زیرپایی اداری , خرج کار , مبلمان و صنایع چوب , اون توستر , جای مسواک و خمیر دندان , مارک لباس , کفی صندلی , تهویه، سرمایش و گرمایش , گوشت کوب برقی , جای دستمال توالت و حوله ای , تور و نوار دوخت , لوازم دوخت و دوز , بخار پز، آرام پز و هوا‌پز برقی , فلاش تانک و لوازم جانبی , کشبافت , حمام و دستشویی , خردکن و آسیاب برقی , سطل و برس توالت , برچسب و استیکر لباس , گریل و ساندویچ ساز , روکش و کاور توالت فرنگی , سر دوشی لباس , باربیکیو و کباب‌پز , لوازم لباس زیر , آفتابه , زودپز , لوازم جانبی توالت فرنگی , میوه خشک کن , سایر لوازم دستشویی , توستر نان , کیک و نان و پیتزا پز برقی , پاپ کورن ساز , سبزی خردکن , سایر لوازم برقی آشپزخانه , گریل , مشعل گازی , سرویس قابلمه , نخ , میوه و گل آرایی مراسم , گل ماشین عروس , گل ولنتاین , پک گل و کیک , ماشین لباسشویی , مشعل گازوئیل سوز , ماهیتابه , سوزن , ماشین ظرفشویی , قیچی خیاطی , مشعل دوگانه / سه‌گانه سوز , قابلمه تکی , جارو برقی , انواع پارچه , مشعل مازوت سوز , ظروف گریل , جارو شارژی , قطعات مشعل , الگوی دوخت , ظروف فر , بخارشوی , ماهیتابه رژیمی , اپل لباس , جارو رباتیک , دیگ سنگی و سفالی , لایی خیاطی , بشکاف خیاطی , اتو , قزن لباس , متر خیاطی , مینی واش , نظافت لباس , صابون و مل خیاطی , کاغذ الگو , پتو شور , نظافت کفش , زانفیکس , رولت خیاطی , دستگاه شیشه پاک کن , لوازم اتو , نوار اریب , ابزار شستشو و نظافت , نوار گان , سایر ابزار و لوازم شستشو و نظافت , کاپ سینه , لوازم یدکی جارو برقی، جاروشارژی و بخارشوی , کاربن خیاطی , لوازم یدکی ماشین لباسشویی و ظرفشویی , کش خیاطی , زیپ , خدمات نظافت , روبان , جعبه خیاطی , دکمه لباس , قالب دکمه , دستگاه پرس دکمه , سرویس غذاخوری , کالای خواب , آباژور و چراغ خواب , ظروف پذیرایی , پارچ و لیوان , ملحفه و روتختی 1 , تجهیزات کمد دیواری , دیس و بشقاب , نگهدارنده درب اتاق , سایر اتاق خواب , قاشق، چنگال و کارد , استکان و فنجان , ماگ , قندان , آبلیمو خوری و سس خوری , شکر پاش , نمکدان و فلفل پاش , کاسه و پیاله , سوفله خوری , اردو خوری و رولت خوری , بستنی خوری , زیتون خوری , سالاد خوری , ظرف عسل و مربا , کره و پنیر خوری , سوپ خوری , ست چاقو کیک , انگاره استکان و لیوان , سطل یخ و یخدان پذیرایی , سایر ظروف سرو و پذیرایی , مبلمان منزل , تهیه و سرو قهوه , تخت خواب , تهیه و سرو چای , آبمیوه گیری , دمنوش ساز , مبلمان و لوازم اداری , بستنی ساز , مخلوط کن برقی , کلمن پذیرایی , پمپ نوشیدنی , میز , صندلی , آبمیوه گیری دستی , سایر نوشیدنی ساز , سایر صنایع چوب , قطعات و لوازم جانبی صنایع چوبی , تهویه مطبوع , همزن دستی , گرمایشی و موتورخانه , رنده و خردکن دستی , آسیاب دستی , استخر، سونا و جکوزی , کمپرسور برودتی , خشک کن دستی , فن سانتریفیوژ , پوست کن , قطعات و لوازم جانبی کمپرسور , سرویس ابزار آشپزخانه , کپسول گاز , سرویس چاقوی آشپزخانه , ساطور , کندانسور برودتی , انبر آشپزی و سرو , شیلنگ گاز , سایر لوازم تهویه، سرمایش و گرمایش , پیتزا بر , چاقو تیزکن , کیسه و صافی چای , هاون و آسیاب سنگی , الک آشپزی , کباب و همبرگر زن خانگی , قندشکن , هسته شکن , سیر له کن , بیفتک و گوشت کوب دستی , دلمه پیچ , پوره کن , درب بازکن قوطی و کنسرو , چرخ گوشت دستی , برس و لیسک آشپزی , قلاب آویز , چاپستیک , پولک گیر , فلافل زن دستی , سایر ابزار آشپزخانه , انبر آشپزی , چرخ خیاطی و ریسندگی , میز تلویزیون , ظروف نگهداری و حمل غذا , دیگ چدنی , میز کامپیوتر و تحریر , ظروف نگهداری , دیگ فولادی , لوازم خرازی , آبکش، لگن و سبد , آینه کنسول , دیگ چگالشی , لوازم و قطعات دستگاه دوخت و دوز , کفگیر و ملاقه , دیگ بخار , میز گیمینگ , تزئینات لباس , میز جلو مبلی و عسلی , تخته کار آشپزخانه , قطعات دیگ , ملزومات خیاطی , قفسه و نظم دهنده آشپزخانه , میز کانتر , میز مکش , فندک آشپزخانه , میز بار و کافی بار , دستگاه لایی چسبان , میز و صندلی آرایش , قیچی آشپزخانه , لوازم عروسک سازی , پاتختی , سفره , کارگاه گلدوزی , میز تلفن , آبچکان و جاظرفی آشپزخانه , دستگاه پرس چاپ حرارتی , سایر ابزار و لوازم جانبی دوخت و دوز , میز اتو و چرخ خیاطی , جای قاشق و چنگال , زیر لیوانی و زیر قابلمه‌ای , سایر میز , دستمال سفره و سفره پاک‌کن , دستگیره و دستکش آشپزخانه , شیشه ویال و جار , درب قابلمه , قالب پیتزا , کاغذ سرخ کن , اسکوپ بستنی , کاغذ و توری روغن گیر , آبگیر سینک , سبد پیک نیک , قالب یخ , آرام ریز , توری و سبد سرخ کن , زعفران دم کن , وارمر قوری , درپوش ظروف , صافی و تفاله گیر سینک , جای سیخ کباب , زیر قاشقی و استند کفگیر , شعله پخش کن , قیف آشپزخانه , توری و سبد بخارپز , یخ مصنوعی , استند و پایه قابلمه , خوشبو کننده هوا , منبع تحت فشار , حوله , ابزار قنادی , تمپر قهوه , پرتافیلتر , لوازم حمام , منبع کویل دار , صندلی غذاخوری و رستورانی , صندلی گیمینگ , منبع دوجداره , مواد اولیه شیرینی پزی , لوازم دستشویی , منبع انبساط , صندلی کانتر و اپنی , قطعات منبع آب گرم , صندلی آزمایشگاهی و پزشکی , صندلی انتظار , صندلی آرایشگاهی , صندلی غذاخوری کودک , صندلی آمفی تئاتر , نیمکت , میز و صندلی نماز , صندلی راک , صندلی و تخت استخری , میز و صندلی تاشو و مسافرتی , سایر صندلی , رگال لباس و لوازم جانبی , نظم دهنده و ارگانایزر , چوب لباسی و آویز لباس , پرزگیر لباس , کاور لباس , سبد لباس چند منظوره , کیسه وکیوم , محفظه و کیسه شستشوی لباس , خوشبو کننده لباس , آب انارگیری , واکس و براق کننده کیف و کفش , گل طبیعی , دستگاه واکس زن , گیاه بنسای , نظم دهنده و باکس کفش , گیاه ساکولنت , پاشنه کش , گیاه کاکتوس , برس و فرچه کفش , گیاه سانسوریا , کفی و زیره کفش , گیاه بنجامین , بوگیر کفش , گیاه دیفن باخیا , کاور کفش , گیاه بابا آدم , گیاه آنتوریوم , خشک کن کفش , گیاه آگلونما , گیاه زاموفیلیا , گیاه یوکا , گیاه فیکوس , گیاه برگ انجیری , گیاه پوتوس , گیاه ارکیده , گیاه شامادورا , گیاه دراسنا , گیاه لیندا , گیاه بامبو , گیاه شفلرا , گیاه حسن یوسف , گیاه برگ عبایی , گیاه بنفشه آفریقایی , گیاه گندمی , گیاه کوکداما , گیاه نخل اریکا , گیاه کروتون , گیاه پاچیرا , تراریوم , گل خشک , سایر گیاهان آپارتمانی , ترمز فرش و پله , دیوار کوب , کوسن , محافظ ریشه فرش , جعبه پذیرایی , تجهیزات بافت فرش , سطل و جای دستمال کاغذی , تزئینات عید نوروز , تزئینات کریسمس , کاور و شال مبل , مگنت , آلات موسیقی دکوری و مینیاتوری , جعبه و گوی موزیکال , زیر سیگاری و جای سیگار , اسپند سوز , اسانس و اسانس سوز , ملزومات پرده , اسنوفر , بک دراپ , عود و جاعودی , بخور معطر , چراغ نفتی , میوه مصنوعی , حلقه و جای دستمال سفره , کاسه تبتی و لوازم جانبی , آبنما , سایر لوازم تزئینی منزل , سطل زباله , تی و سرویس زمین شوی , بند رخت و خشک‌کن لباس , گیره آویز لباس , جارو و خاک انداز , اسکاچ و سیم ظرفشویی , ابر و دستمال نظافت , فرچه و برس شستشو , پاکت جارو برقی , جای اسکاچ و مایع ظرفشویی , ترولی حمل , اتو سرد و سنگی , دستکش نظافت , جای نایلون , ضربه گیر درب و دیوار , آویز و نگهدارنده کیسه زباله , مگس کش , دستگاه حشره‌کش , ظرف و جای پودر شوینده , کمد، بوفه و قفسه دیواری , ست میز و صندلی غذاخوری , دراور و کشو , جاکفشی و جالباسی , کتابخانه , پشتی و سرویس شاه‌نشین , چهارپایه , تاب ریلکسی , تخت سنتی , پاراوان و پارتیشن , لوازم خانگی"

        user_content.append({
            "type": "text",
            "text": (
                    "CANDIDATE TITLES (Persian) — coarse_label MUST be chosen from this list.\n"
                    + ALL_TITLES
            )
        })

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        completion = await client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.0,
        )

        raw = completion.choices[0].message.content.strip() if completion.choices else ""
        data: Dict[str, str] | None = None
        try:
            data = json.loads(raw)
        except Exception:
            lines = [re.sub(r'[\"“”«»]+', '', ln).strip() for ln in raw.splitlines() if ln.strip()]
            if len(lines) >= 2:
                data = {"coarse_label": lines[0], "fine_label": lines[1]}
            elif len(lines) == 1:
                data = {"coarse_label": lines[0], "fine_label": lines[0]}
            else:
                data = None

        if not data:
            return None

        for k in ("coarse_label", "fine_label"):
            if k in data and isinstance(data[k], str):
                data[k] = re.sub(r"[\"'“”«».!؟?\s]+$", "", data[k]).strip() or "unknown"

        coarse = data.get("coarse_label") or "unknown"
        fine = data.get("fine_label") or coarse
        return {"coarse_label": coarse, "fine_label": fine}

    except Exception as e:
        logging.exception(f"classify_main_object_from_image_query failed: {e}")
        return None


# -------------------- Scenario 7: Suggest object based on the product in an image --------------------
_SC7_PAT = re.compile(r"(محصول|بیس).*(مرتبط|مناسب).*تصویر|بده|پیشنهاد", re.I)


def is_best_base_from_image_query(text: str) -> bool:
    t = (text or "").strip()
    return bool(t and _SC7_PAT.search(t))


def _norm_fa(s: str) -> str:
    return normalize_fa(s or "").strip()


def _pick_category_id_from_name(name: str, categories_df: pd.DataFrame) -> int | None:
    if categories_df is None or categories_df.empty or not name:
        return None

    title_cols = [c for c in ["title", "id", "parent_id"] if c in categories_df.columns]
    if not title_cols or "id" not in categories_df.columns:
        return None

    q = _norm_fa(name)

    for col in title_cols:
        coln = f"__norm_{col}"
        if coln not in categories_df.columns:
            categories_df[coln] = categories_df[col].astype(str).map(_norm_fa)

        exact = categories_df[categories_df[coln] == q]
        print(exact)
        if not exact.empty:
            return int(exact["id"].iloc[0])

    for col in title_cols:
        coln = f"__norm_{col}"
        sub = categories_df[categories_df[coln].str.contains(q, na=False)]
        if not sub.empty:
            return int(sub["id"].iloc[0])

    try:
        from rapidfuzz import process, fuzz
        col = title_cols[0]
        names = categories_df[col].astype(str).tolist()
        best = process.extractOne(name, names, scorer=fuzz.token_set_ratio)
        if best and best[1] >= 80:
            idx = best[2]
            return int(categories_df.iloc[idx]["id"])
    except Exception:
        pass

    return None


def _pick_one_base_key_for_category(category_id: int | str, products_df: pd.DataFrame) -> str | None:
    if products_df is None or products_df.empty or "category_id" not in products_df.columns:
        return None
    if "random_key" not in products_df.columns:
        return None

    sub = products_df[products_df["category_id"].astype(str) == str(category_id)]
    if sub.empty:
        return None

    return str(sub["random_key"].iloc[0])


async def suggest_best_base_from_image(
        full_request_payload: dict,
        categories_df: pd.DataFrame,
        products_df: pd.DataFrame,
) -> str | None:

    label = await classify_main_object_from_image_query(full_request_payload)
    print(label)
    if not label:
        return None

    cat_id = _pick_category_id_from_name(label, categories_df)
    if cat_id is None:
        return None

    return _pick_one_base_key_for_category(cat_id, products_df)


def is_image_to_base_query(query: str) -> bool:

    q = normalize_fa(query or "")
    hints = [
        "محصول مرتبط", "بیس مربوط", "معادل محصول", "مرتبط با تصویر",
        "بهترین بیس", "بیس این عکس", "random key", "رندوم کی", "base key"
    ]
    return any(h in q for h in hints)


def fuzzy_find_base_by_persian_name(name: str, products_df: pd.DataFrame, *, min_score: int = 80) -> Optional[str]:

    try:
        if not name or products_df is None or products_df.empty:
            return None

        if 'persian_name' not in products_df.columns or 'random_key' not in products_df.columns:
            return None

        from rapidfuzz import process, fuzz

        candidates = products_df['persian_name'].astype(str).fillna("").tolist()
        query_norm = normalize_fa(name)
        cand_norm = [normalize_fa(x) for x in candidates]

        match = process.extractOne(
            query_norm,
            cand_norm,
            scorer=fuzz.token_set_ratio
        )
        if not match:
            return None

        matched_text, score, idx = match
        if score < min_score:
            match2 = process.extractOne(query_norm, cand_norm, scorer=fuzz.partial_ratio)
            if not match2 or match2[1] < min_score:
                return None
            idx = match2[2]

        rk = products_df.iloc[idx]['random_key']
        return str(rk) if pd.notna(rk) else None

    except Exception as e:
        logging.exception(f"fuzzy_find_base_by_persian_name failed: {e}")
        return None


# --- Category matching ---

_CATEGORY_TITLE_CANDIDATES = ["title", "name", "fa_title", "fa_name", "category_title", "category_name"]

_GENERIC_PHRASE_PATTERNS = [
    r"چند\s*کاره",
    r"خوب",
    r"با\s*کیفیت",
    r"کیفیت\s*بالا",
    r"قیمت\s*مناسب",
    r"به\s*صرفه",
]


def strip_generic_modifiers(name: str) -> str:
    s = normalize_fa(name or "")
    for pat in _GENERIC_PHRASE_PATTERNS:
        s = re.sub(rf"(?<!\S){pat}(?!\S)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _iter_category_titles_norm(categories_df: pd.DataFrame):
    if categories_df is None or categories_df.empty or "id" not in categories_df.columns:
        return
    for col in _CATEGORY_TITLE_CANDIDATES:
        if col in categories_df.columns:
            col_norm = categories_df[col].astype(str).map(normalize_fa)
            for t, cid in zip(col_norm, categories_df["id"]):
                tn = (t or "").strip()
                if tn:
                    yield tn, int(cid)


def _pick_category_id_from_name_exact(name: str, categories_df: pd.DataFrame) -> int | None:

    if not name or categories_df is None or categories_df.empty or "id" not in categories_df.columns:
        return None
    q = normalize_fa(name).strip()
    if not q:
        return None
    for tn, cid in _iter_category_titles_norm(categories_df):
        if tn == q:
            return cid
    return None


def is_generic_product_phrase(name: str, categories_df: pd.DataFrame) -> bool:

    s = normalize_fa(name or "")
    if not s:
        return False

    if extract_model_token(s) or extract_catalog_code_any(s) or _LATIN_RE.search(s):
        return False

    if _pick_category_id_from_name_exact(s, categories_df) is not None:
        return True

    s2 = strip_generic_modifiers(s)
    if s2 and _pick_category_id_from_name_exact(s2, categories_df) is not None:
        return True

    return False



def _load_keylist(path: str | None) -> set[str]:

    keys: set[str] = set()
    if not path:
        return keys
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                keys.add(s)
    except Exception as e:
        logging.info("Scenario7: keylist not loaded from %r (%s)", path, e)
    return keys


def _filter_products_by_category(products_df: pd.DataFrame, category_id: int | str) -> pd.DataFrame:
    if products_df is None or products_df.empty or "category_id" not in products_df.columns:
        return pd.DataFrame(columns=products_df.columns if products_df is not None else [])
    return products_df.loc[products_df["category_id"].astype(str) == str(category_id)].copy()


def _apply_keylist_filter(df: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    if not keys or df.empty or "random_key" not in df.columns:
        return df
    mask = df["random_key"].astype(str).isin(keys)
    sub = df.loc[mask].copy()
    return sub


def choose_base_by_labels(
        labels: dict,
        categories_df: pd.DataFrame,
        products_df: pd.DataFrame,
        *,
        keylist_path: str | None = None,
        strict_keylist: bool = False,
) -> str | None:

    if not labels:
        return None

    coarse = (labels.get("coarse_label") or "").strip()
    fine = (labels.get("fine_label") or "").strip() or coarse
    if not coarse:
        return None

    cat_id = _pick_category_id_from_name_exact(coarse, categories_df) or _pick_category_id_from_name(coarse,
                                                                                                     categories_df)
    if cat_id is None:
        return None

    subset = _filter_products_by_category(products_df, cat_id)
    logging.info(f"--- category filter: '{len(subset)}' ---")
    if subset.empty:
        return None

    keys = _load_keylist(keylist_path or getattr(settings, "scenario7_keylist_path", "label.txt"))
    if keys:
        narrowed = _apply_keylist_filter(subset, keys)
        logging.info(f"--- keylist filter txt: '{len(narrowed)}' ---")
        if not narrowed.empty:
            subset = narrowed
        elif strict_keylist:
            return None

    key = fuzzy_find_base_by_persian_name(fine, subset) or fuzzy_find_base_by_persian_name(coarse, subset)
    if key:
        return key

    try:
        return str(subset["random_key"].iloc[0])
    except Exception:
        return None