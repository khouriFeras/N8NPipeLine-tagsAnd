#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Translation and SEO Generator for N8N
Unified script that handles any product type with configurable prompts and settings.
Designed for use with N8N automation workflows.

Author: Your Name
Created: September 2024
"""

import os
import re
import json
import argparse
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI

# ---------- Logger Setup ----------
def setup_logger(name: str = "universal_translator", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with custom formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()

# ---------- Utility Functions ----------
def clean_ws(s: Optional[str]) -> str:
    """Clean whitespace from string."""
    if not isinstance(s, str): 
        return ""
    return re.sub(r"\s+", " ", s).strip()

def truncate_chars(s: str, max_chars: int) -> str:
    """Truncate string to max_chars with ellipsis."""
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + "…")

def dedupe_key(*parts: str) -> str:
    """Generate deduplication key from parts."""
    base = "|".join([clean_ws(p).lower() for p in parts])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def sanitize_value(value: Any) -> str:
    """Sanitize value for processing."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = clean_ws(text)
    lowered = text.lower()
    placeholders = {"nan", "none", "null", "n/a", "na", "-", "--", "غير موجود", "غير موجدود", "غير متوفر"}
    return "" if lowered in placeholders else text

def normalize_units(text: str) -> str:
    """Normalize common units to uppercase."""
    s = text or ""
    
    # All common units - let the AI handle context-specific normalization
    s = re.sub(r"(?i)\b(\d+)\s*mm\b", r"\1MM", s)
    s = re.sub(r"(?i)\b(\d+)\s*cm\b", r"\1CM", s)
    s = re.sub(r"(?i)\b(\d+)\s*inch\b", r"\1INCH", s)
    s = re.sub(r"(?i)\b(\d+)\s*w\b", r"\1W", s)
    s = re.sub(r"(?i)\b(\d+)\s*v\b", r"\1V", s)
    s = re.sub(r"(?i)\b(\d+)\s*amp\b", r"\1AMP", s)
    s = re.sub(r"(?i)\b(\d+)\s*ah\b", r"\1AH", s)
    s = re.sub(r"(?i)\b(\d+)\s*ml\b", r"\1ML", s)
    s = re.sub(r"(?i)\b(\d+)\s*l\b", r"\1L", s)
    s = re.sub(r"(?i)\b(\d+)\s*kg\b", r"\1KG", s)
    s = re.sub(r"(?i)\b(\d+)\s*cc\b", r"\1CC", s)
    s = re.sub(r"(?i)\b(\d+)\s*psi\b", r"\1PSI", s)
    s = re.sub(r"(?i)\b(\d+)\s*bar\b", r"\1BAR", s)
    
    return s

def ensure_brand_suffix(title: str, brand: str) -> str:
    """Ensure brand appears as 'من BRAND' at the end."""
    t = clean_ws(title)
    b = clean_ws(brand)
    if not b:
        return t
    suffix = f"من {b}"
    if suffix not in t:
        t = f"{t} {suffix}".strip()
    # Remove duplicates
    t = re.sub(rf"\s*\bمن\s+{re.escape(b)}\b(?:\s*من\s+{re.escape(b)}\b)+", suffix, t)
    return t

def create_seo_handle(title: str) -> str:
    """Create SEO-friendly handle from Arabic title."""
    if not title:
        return ""
    
    # Clean the title
    handle = clean_ws(title)
    
    # Remove HTML tags if any
    handle = re.sub(r'<[^>]+>', '', handle)
    
    # Replace spaces with hyphens for URL-friendly format
    handle = re.sub(r'\s+', '-', handle).strip()
    
    # Remove any special characters except hyphens
    handle = re.sub(r'[^\w\-]', '', handle)
    
    # Clean up multiple hyphens
    handle = re.sub(r'-+', '-', handle)
    handle = handle.strip('-')
    
    return handle

# ---------- Default Base Prompts (Generic like translate_to_arVer2.py) ----------
DEFAULT_SYSTEM_PROMPT = (
    "أنت خبير تسويق ومنتجات يتقن العربية الفصحى المستخدمة تجارياً.\n"
    "المهمة: ترجمة/تلخيص بيانات المنتجات من الإنجليزية إلى العربية وتهيئتها لقالب ثابت.\n"
    "القواعد: لا تُترجم العلامة التجارية/الموديل؛ حافظ على الأرقام والوحدات؛ لا تختلق؛ JSON فقط بالمفاتيح المطلوبة.\n"
)

DEFAULT_META_TITLE_PROMPT = """أنت خبير SEO متخصص في التجارة الإلكترونية.
مهمتك: كتابة Meta Title قصير وجذاب لمنتج باللغة العربية الفصحى.

القواعد:
- لا يتجاوز 60 حرفًا.
- يحتوي على الكلمة المفتاحية الأساسية (اسم المنتج + الفئة إن وُجدت).
- واضح ويشجع على النقر.
- بدون حشو أو تكرار غير ضروري.

الإدخال:
- اسم المنتج: {product_name}
- الفئة: {category}

المطلوب:
أعطني سطرًا واحدًا فقط يمثل الـ Meta Title."""

DEFAULT_META_DESC_PROMPT = """أنت خبير SEO متخصص في التجارة الإلكترونية.
مهمتك: كتابة Meta Description تسويقي وجذاب لمنتج باللغة العربية الفصحى.

القواعد:
- لا يتجاوز 160 حرفًا.
- يتضمن الفوائد/المزايا الأساسية المستقاة من الوصف.
- يتضمن دعوة للفعل (مثل: اطلب الآن، تسوق الآن).
- أسلوب مهني، بدون مبالغة فارغة أو تكرار.

الإدخال:
- اسم المنتج: {product_name}
- الفئة: {category}
- المميزات/الوصف المختصر: {features}

المطلوب:
أعطني سطرًا واحدًا فقط يمثل الـ Meta Description."""

DEFAULT_PRODUCT_TITLE_PROMPT = """اكتب عنواناً جذاباً وواضحاً باللغة العربية لمنتج يُعرض على متجر إلكتروني.
يجب أن يتضمن:
- اسم المنتج الأساسي.
- المواصفات الرئيسية (مثل الحجم، الدقة، السعة، الموديل) إن وُجدت.
- المزايا الفريدة (مثل التقنيات المدعومة أو الميزات الخاصة) إن وُجدت.
- العلامة التجارية (يجب ذكرها دائماً).
الطول المطلوب: بين 50 و80 حرفاً.
يجب أن يكون العنوان محسّنًا لتحسين محركات البحث (SEO) باستخدام الكلمات المفتاحية الأساسية.
مهم جداً: استخدم فقط المعلومات المتوفرة أدناه دون أي زيادة أو اختلاق.

المعلومات المتاحة:
- الاسم: {name_ar}
- العلامة التجارية: {brand}
- مواصفات رئيسية: {specs}
- مزايا/ميزات: {benefits}

المطلوب:
أعطني سطراً واحداً فقط يمثل العنوان العربي ضمن 50–80 حرفاً، بدون أي رموز تعبيرية وبدون علامات ترقيم مبالغ فيها."""

# ---------- Translation Template ----------
USER_TEMPLATE = """\
Convert and structure the following product. Translate EN -> AR for all natural-language text EXCEPT brand and model codes.

Return ONLY valid JSON with these exact keys:
- name_ar
- name_en
- brand
- benefits
- specs
- short_overview

Constraints:
- Keep facts grounded in input text. Do not invent specs.
- If a field is missing in input, omit it from bullets/specs rather than guessing.
- Focus on product functionality, durability, and use cases.

INPUT:
- name_en: {name_en}
- brand: {brand}
- description_en: {description}
"""

# ---------- JSON Schema ----------
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ProductArabicV1",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name_ar": {"type": "string"},
                "name_en": {"type": "string"},
                "brand": {"type": "string"},
                "benefits": {"type": "array", "items": {"type": "string"}},
                "specs": {"type": "array", "items": {"type": "string"}},
                "short_overview": {"type": "string"}
            },
            "required": ["name_ar", "name_en", "brand", "benefits", "specs", "short_overview"]
        }
    }
}

DISALLOW_TEMP = {"gpt-5-mini", "gpt-5-nano"}

# ---------- Flexible Prompt Builders ----------
def build_user_prompt(name_en: str, brand: str, description: str, cap: int, custom_system_prompt: str = None) -> str:
    """Build user prompt for translation with optional custom system prompt."""
    desc = truncate_chars(clean_ws(description), cap)
    return USER_TEMPLATE.format(
        name_en=clean_ws(name_en),
        brand=clean_ws(brand),
        description=desc,
    )

def build_meta_title_prompt(product_name: str, feature: str, custom_prompt: str = None) -> str:
    """Build meta title prompt with optional custom template."""
    if custom_prompt:
        return custom_prompt.format(
            product_name=product_name,
            category=feature,
            feature=feature
        )
    
    return DEFAULT_META_TITLE_PROMPT.format(
        product_name=product_name,
        category=feature
    )

def build_meta_desc_prompt(product_name: str, arabic_desc: str, features_text: str, cap: int, custom_prompt: str = None) -> str:
    """Build meta description prompt with optional custom template."""
    if custom_prompt:
        return custom_prompt.format(
            product_name=product_name,
            category=arabic_desc,
            features=truncate_chars(clean_ws(features_text), 100),
            arabic_desc=truncate_chars(clean_ws(arabic_desc), cap)
        )
    
    return DEFAULT_META_DESC_PROMPT.format(
        product_name=product_name,
        category=arabic_desc,
        features=truncate_chars(clean_ws(features_text), 100)
    )

def build_product_title_prompt(name_ar: str, brand: str, specs: List[str], benefits: List[str], custom_prompt: str = None) -> str:
    """Build product title prompt with optional custom template."""
    specs_txt = " | ".join([clean_ws(s) for s in specs[:4]])
    bens_txt = " | ".join([clean_ws(b) for b in benefits[:3]])
    
    if custom_prompt:
        return custom_prompt.format(
            name_ar=clean_ws(name_ar),
            brand=clean_ws(brand),
            specs=clean_ws(specs_txt),
            benefits=clean_ws(bens_txt)
        )
    
    return DEFAULT_PRODUCT_TITLE_PROMPT.format(
        name_ar=clean_ws(name_ar),
        brand=clean_ws(brand),
        specs=clean_ws(specs_txt),
        benefits=clean_ws(bens_txt)
    )

# ---------- Content Composition ----------
def compose_single_block(obj: Dict[str, Any], sku: str = "", model: str = "", override_name_ar: Optional[str] = None) -> str:
    """Compose Arabic description block."""
    name_ar = (override_name_ar if override_name_ar is not None else obj.get("name_ar", "")).strip()
    name_en = obj.get("name_en", "").strip()
    brand = obj.get("brand", "").strip()
    benefits = [f"• {str(b).strip()}" for b in obj.get("benefits", []) if str(b).strip()]
    specs = [str(s).strip() for s in obj.get("specs", []) if str(s).strip()]
    overview = obj.get("short_overview", "").strip()

    parts = [
        f"اسم المنتج بالعربي: {name_ar}" if name_ar else "",
        f"اسم المنتج بالإنجليزي: {name_en}" if name_en else "",
        f"الماركة: {brand}" if brand else "",
        (f"الموديل/الكود: {model}" if model else ""),
        (f"كود المنتج: {sku}" if sku else ""),
    ]
    if benefits:
        parts += ["", "وصف المنتج وفوائده:", *benefits]
    if specs:
        parts += ["", "المواصفات الفنية:", *specs]
    if overview:
        parts += ["", "شرح موجز للخواص العامة:", overview]
    return "\n".join([p for p in parts if p]).strip()

# ---------- I/O Functions ----------
def read_frame(path: Path) -> pd.DataFrame:
    """Read Excel or CSV file."""
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}: 
        return pd.read_excel(path)
    if suf == ".csv": 
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension: {path.suffix}")

def write_frame(df: pd.DataFrame, path: Path):
    """Write Excel or CSV file."""
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        try:
            df.to_excel(path, index=False, engine="xlsxwriter")
        except Exception:
            df.to_excel(path, index=False)
    elif suf == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported output extension: {path.suffix}")

def extract_json(s: str) -> str:
    """Extract JSON from string."""
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)

def is_schema_unsupported(exc: Exception) -> bool:
    """Check if exception is due to unsupported schema."""
    s = str(exc).lower()
    return ("response_format" in s and ("unsupported" in s or "not supported" in s))

# ---------- Main Translator Class ----------
class UniversalTranslator:
    """Universal translator with custom prompt support."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 temperature: Optional[float] = None,
                 desc_cap: int = 900,
                 use_schema: bool = True,
                 timeout: float = 60.0,
                 custom_system_prompt: str = None,
                 custom_meta_title_prompt: str = None,
                 custom_meta_desc_prompt: str = None,
                 custom_product_title_prompt: str = None):
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.desc_cap = desc_cap
        self.use_schema = use_schema
        self.custom_system_prompt = custom_system_prompt
        self.custom_meta_title_prompt = custom_meta_title_prompt
        self.custom_meta_desc_prompt = custom_meta_desc_prompt
        self.custom_product_title_prompt = custom_product_title_prompt
        self._send_temp = (temperature is not None) and (model not in DISALLOW_TEMP)
        self._temperature = temperature
        if (temperature is not None) and (model in DISALLOW_TEMP):
            logger.info(f"Model '{model}' ignores explicit temperature; using provider default.")

    async def _create(self, *, messages, response_format=None, use_temp=True):
        """Create completion with retry logic."""
        kwargs = dict(model=self.model, messages=messages)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if use_temp and self._send_temp:
            kwargs["temperature"] = self._temperature
        return await self.client.chat.completions.create(**kwargs)

    @retry(reraise=True, stop=stop_after_attempt(6), wait=wait_exponential(1, 1, 60),
           retry=retry_if_exception_type(Exception))
    async def translate_one(self, name_en: str, brand: str, description: str) -> Dict[str, Any]:
        """Translate a single product with custom or default prompts."""
        system_prompt = self.custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(name_en, brand, description, self.desc_cap, self.custom_system_prompt)},
        ]
        
        if self.use_schema:
            try:
                resp = await self._create(messages=messages, response_format=JSON_SCHEMA, use_temp=True)
                return json.loads(resp.choices[0].message.content.strip())
            except Exception as e:
                if not is_schema_unsupported(e):
                    raise
        try:
            resp = await self._create(messages=messages, response_format={"type": "json_object"}, use_temp=True)
            return json.loads(resp.choices[0].message.content.strip())
        except Exception:
            resp = await self._create(messages=messages, response_format=None, use_temp=False)
            return json.loads(extract_json(resp.choices[0].message.content.strip()))

    @retry(reraise=True, stop=stop_after_attempt(6), wait=wait_exponential(1, 1, 60),
           retry=retry_if_exception_type(Exception))
    async def gen_meta_title(self, product_name: str, feature: str) -> str:
        """Generate meta title with custom or default prompt."""
        messages = [{"role": "user", "content": build_meta_title_prompt(product_name, feature, self.custom_meta_title_prompt)}]
        resp = await self._create(messages=messages, response_format=None, use_temp=True)
        return clean_ws(resp.choices[0].message.content)

    @retry(reraise=True, stop=stop_after_attempt(6), wait=wait_exponential(1, 1, 60),
           retry=retry_if_exception_type(Exception))
    async def gen_meta_desc(self, product_name: str, arabic_desc: str, features_text: str) -> str:
        """Generate meta description with custom or default prompt."""
        messages = [{"role": "user", "content": build_meta_desc_prompt(product_name, arabic_desc, features_text, self.desc_cap, self.custom_meta_desc_prompt)}]
        resp = await self._create(messages=messages, response_format=None, use_temp=True)
        return clean_ws(resp.choices[0].message.content)

    @retry(reraise=True, stop=stop_after_attempt(6), wait=wait_exponential(1, 1, 60),
           retry=retry_if_exception_type(Exception))
    async def gen_product_title(self, name_ar: str, brand: str, specs: List[str], benefits: List[str]) -> str:
        """Generate product title with custom or default prompt."""
        prompt = build_product_title_prompt(name_ar, brand, specs, benefits, self.custom_product_title_prompt)
        messages = [{"role": "user", "content": prompt}]
        resp = await self._create(messages=messages, response_format=None, use_temp=True)
        raw = clean_ws(resp.choices[0].message.content)
        raw = normalize_units(raw)
        raw = ensure_brand_suffix(raw, brand)
        return raw

# ---------- Main Processing Function ----------
async def process_excel_file(
    input_path: Path,
    output_path: Path,
    name_col: str = "Title",
    brand_col: str = "Brand",
    desc_col: str = "Body (HTML)",
    cat_col: Optional[str] = None,
    sku_col: Optional[str] = None,
    model_col: Optional[str] = None,
    api_key: str = "",
    model: str = "gpt-4o-mini",
    workers: int = 5,
    checkpoint_every: int = 100,
    sample: int = 0,
    desc_cap: int = 900,
    temperature: Optional[float] = None,
    no_schema: bool = False,
    timeout: float = 60.0,
    # Custom prompt parameters for N8N flexibility
    custom_system_prompt: str = None,
    custom_meta_title_prompt: str = None,
    custom_meta_desc_prompt: str = None,
    custom_product_title_prompt: str = None,
    product_context: str = "منتجات متنوعة"  # Generic context instead of fixed product_type
) -> Dict[str, Any]:
    """
    Process Excel file with translation and SEO generation.
    
    Returns:
        Dict with processing statistics and results.
    """
    
    logger.info(f"Processing {input_path} with context: {product_context}")
    logger.info(f"Custom prompts: System={bool(custom_system_prompt)}, MetaTitle={bool(custom_meta_title_prompt)}, MetaDesc={bool(custom_meta_desc_prompt)}, ProductTitle={bool(custom_product_title_prompt)}")
    
    # Read input file
    df = read_frame(input_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Check required columns
    required_cols = [name_col, desc_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}. Present: {list(df.columns)}")
    
    # Check optional columns
    optional_cols = [brand_col, cat_col, sku_col, model_col]
    for col_name, col_var in zip(["brand", "category", "sku", "model"], optional_cols):
        if col_var and col_var not in df.columns:
            logger.warning(f"{col_name.title()} column '{col_var}' not found; continuing without it.")
            if col_name == "brand":
                brand_col = None
            elif col_name == "category":
                cat_col = None
            elif col_name == "sku":
                sku_col = None
            elif col_name == "model":
                model_col = None
    
    # Sample data if requested
    if sample > 0:
        df = df.head(sample).copy()
        logger.info(f"Processing sample of {sample} rows")
    
    df = df.reset_index(drop=False).rename(columns={"index": "__row__"})
    
    # Initialize translator with custom prompts
    translator = UniversalTranslator(
        api_key=api_key,
        model=model,
        temperature=temperature,
        desc_cap=desc_cap,
        use_schema=(not no_schema),
        timeout=timeout,
        custom_system_prompt=custom_system_prompt,
        custom_meta_title_prompt=custom_meta_title_prompt,
        custom_meta_desc_prompt=custom_meta_desc_prompt,
        custom_product_title_prompt=custom_product_title_prompt
    )
    
    # Caches for deduplication
    cache_translate: Dict[str, Dict[str, Any]] = {}
    cache_title: Dict[str, str] = {}
    cache_desc: Dict[str, str] = {}
    cache_prodtitle: Dict[str, str] = {}
    results: Dict[int, Dict[str, str]] = {}
    
    # Create processing queue
    q: asyncio.Queue[Tuple[int, pd.Series]] = asyncio.Queue()
    for i, row in df.iterrows():
        q.put_nowait((i, row))
    
    pbar = tqdm(total=len(df), desc=f"Processing {product_context} (async x{workers})")
    lock = asyncio.Lock()
    
    async def worker():
        """Worker function for processing products."""
        while True:
            try:
                i, row = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            
            row_id = int(row["__row__"])
            name_en = sanitize_value(row.get(name_col, ""))
            brand = sanitize_value(row.get(brand_col, "")) if brand_col else ""
            desc_en = sanitize_value(row.get(desc_col, ""))
            category = sanitize_value(row.get(cat_col, "")) if cat_col else ""
            sku = sanitize_value(row.get(sku_col, "")) if sku_col else ""
            model_v = sanitize_value(row.get(model_col, "")) if model_col else ""
            
            arabic_block = ""
            meta_title = ""
            meta_description = ""
            product_title_ar = ""
            
            if not (name_en or brand or desc_en):
                # Empty row
                obj = {"name_ar": "", "name_en": name_en, "brand": brand,
                       "benefits": [], "specs": [], "short_overview": ""}
                arabic_block = compose_single_block(obj, sku=sku, model=model_v)
            else:
                # Process with AI
                k_trans = dedupe_key("T", name_en, brand, desc_en, str(desc_cap))
                if k_trans in cache_translate:
                    obj = cache_translate[k_trans]
                else:
                    try:
                        obj = await translator.translate_one(name_en, brand, desc_en)
                    except Exception as e:
                        logger.error(f"[Row {row_id}] translation failed: {e}")
                        obj = {"name_ar": "", "name_en": name_en, "brand": brand,
                               "benefits": [], "specs": [], "short_overview": f"تعذر المعالجة: {e}"}
                    cache_translate[k_trans] = obj
                
                name_ar = obj.get("name_ar", "")
                specs_list = obj.get("specs", []) or []
                benefits_list = obj.get("benefits", []) or []
                overview = obj.get("short_overview", "") or ""
                
                # Generate product title (50-80 chars)
                k_pt = dedupe_key("PT", name_ar, brand, " | ".join(specs_list[:4]), " | ".join(benefits_list[:3]))
                if k_pt in cache_prodtitle:
                    product_title_ar = cache_prodtitle[k_pt]
                else:
                    try:
                        product_title_ar = await translator.gen_product_title(
                            name_ar=name_ar, brand=brand, specs=specs_list, benefits=benefits_list)
                        if not (50 <= len(product_title_ar) <= 80):
                            product_title_ar = truncate_chars(product_title_ar, 80)
                    except Exception as e:
                        logger.error(f"[Row {row_id}] product_title_ar failed: {e}")
                        product_title_ar = ""
                    cache_prodtitle[k_pt] = product_title_ar
                
                # Compose Arabic block
                arabic_block = compose_single_block(
                    obj, sku=sku, model=model_v, 
                    override_name_ar=product_title_ar or name_ar)
                
                # Generate meta title (≤60 chars)
                base_for_meta = product_title_ar or obj.get("name_ar") or name_en
                feature_context = category or product_context
                k_title = dedupe_key("MT", base_for_meta, feature_context)
                if k_title in cache_title:
                    meta_title = cache_title[k_title]
                else:
                    try:
                        meta_title = await translator.gen_meta_title(base_for_meta, feature_context)
                        if len(meta_title) > 60:
                            meta_title = truncate_chars(meta_title, 60)
                    except Exception as e:
                        logger.error(f"[Row {row_id}] meta_title failed: {e}")
                        meta_title = ""
                    cache_title[k_title] = meta_title
                
                # Generate meta description (≤155 chars)
                features_text = "؛ ".join(benefits_list[:3]) or overview or desc_en
                arabic_desc_input = overview or "؛ ".join(benefits_list[:3]) or name_ar
                k_desc = dedupe_key("MD", base_for_meta, arabic_desc_input[:120], features_text[:120])
                if k_desc in cache_desc:
                    meta_description = cache_desc[k_desc]
                else:
                    try:
                        meta_description = await translator.gen_meta_desc(
                            base_for_meta, arabic_desc_input, features_text)
                        if len(meta_description) > 155:
                            meta_description = truncate_chars(meta_description, 155)
                    except Exception as e:
                        logger.error(f"[Row {row_id}] meta_description failed: {e}")
                        meta_description = ""
                    cache_desc[k_desc] = meta_description
            
            # Store results
            async with lock:
                results[row_id] = {
                    "arabic_description": arabic_block,
                    "meta_title": meta_title,
                    "meta_description": meta_description,
                    "product_title_ar": product_title_ar,
                }
                pbar.update(1)
                
                # Save checkpoint
                if checkpoint_every and (len(results) % checkpoint_every == 0):
                    tmp = df.copy()
                    # Translation data columns
                    tmp["arabic description"] = tmp["__row__"].map(lambda r: results.get(r, {}).get("arabic_description", ""))
                    tmp["product_title_ar"] = tmp["__row__"].map(lambda r: results.get(r, {}).get("product_title_ar", ""))
                    
                    # SEO data columns
                    tmp["meta_title"] = tmp["__row__"].map(lambda r: results.get(r, {}).get("meta_title", ""))
                    tmp["meta_description"] = tmp["__row__"].map(lambda r: results.get(r, {}).get("meta_description", ""))
                    
                    # Generate handle for SEO-friendly URLs
                    tmp["handle_ar"] = tmp["product_title_ar"].apply(lambda x: create_seo_handle(x) if x else "")
                    
                    write_frame(tmp.drop(columns=["__row__"]), output_path)
                    logger.info(f"Checkpoint saved at {len(results)} rows -> {output_path}")
    
    # Run workers
    tasks = [asyncio.create_task(worker()) for _ in range(max(1, workers))]
    await asyncio.gather(*tasks)
    pbar.close()
    
    # Final save - append all translation and SEO data to new columns
    final = df.copy()
    
    # Translation data columns
    final["arabic description"] = final["__row__"].map(lambda r: results.get(r, {}).get("arabic_description", ""))
    final["product_title_ar"] = final["__row__"].map(lambda r: results.get(r, {}).get("product_title_ar", ""))
    
    # SEO data columns
    final["meta_title"] = final["__row__"].map(lambda r: results.get(r, {}).get("meta_title", ""))
    final["meta_description"] = final["__row__"].map(lambda r: results.get(r, {}).get("meta_description", ""))
    
    # Generate handle for SEO-friendly URLs
    final["handle_ar"] = final["product_title_ar"].apply(lambda x: create_seo_handle(x) if x else "")
    
    write_frame(final.drop(columns=["__row__"]), output_path)
    
    # Statistics
    successful = sum(1 for r in results.values() if r.get("arabic_description"))
    failed = len(results) - successful
    seo_successful = sum(1 for r in results.values() if r.get("meta_title") and r.get("meta_description"))
    
    stats = {
        "total_processed": len(results),
        "translation_successful": successful,
        "seo_successful": seo_successful,
        "failed": failed,
        "output_file": str(output_path),
        "product_context": product_context,
        "output_columns": [
            "arabic description", "product_title_ar", "meta_title", "meta_description", "handle_ar"
        ],
        "custom_prompts_used": {
            "system": bool(custom_system_prompt),
            "meta_title": bool(custom_meta_title_prompt),
            "meta_desc": bool(custom_meta_desc_prompt),
            "product_title": bool(custom_product_title_prompt)
        }
    }
    
    logger.info(f"Processing complete!")
    logger.info(f"Total: {stats['total_processed']}, Translation Success: {stats['translation_successful']}, SEO Success: {stats['seo_successful']}, Failed: {stats['failed']}")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"New columns added: {', '.join(stats['output_columns'])}")
    
    return stats

# ---------- CLI Interface ----------
def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Universal Translation and SEO Generator for N8N (Generic Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Custom Prompt Support:
  System Prompt Variables: {name_en}, {brand}, {description}
  Meta Title Variables: {product_name}, {category}
  Meta Description Variables: {product_name}, {category}, {features}
  Product Title Variables: {name_ar}, {brand}, {specs}, {benefits}

Example Usage:
  python universal_translation_seo.py --input products.xlsx --output results.xlsx --context "power tools"
  python universal_translation_seo.py --input products.xlsx --output results.xlsx --system-prompt "أنت خبير في الأجهزة الإلكترونية"
        """
    )
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Input Excel/CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output Excel/CSV file path")
    
    # Product context (replaces product-type)
    parser.add_argument("--context", "-c", default="منتجات متنوعة", 
                        help="Product context description (default: منتجات متنوعة)")
    
    # Custom prompts
    parser.add_argument("--system-prompt", help="Custom system prompt for translation")
    parser.add_argument("--meta-title-prompt", help="Custom meta title generation prompt")
    parser.add_argument("--meta-desc-prompt", help="Custom meta description generation prompt")
    parser.add_argument("--product-title-prompt", help="Custom product title generation prompt")
    
    # Column mapping
    parser.add_argument("--name-col", default="Title", help="Product name column")
    parser.add_argument("--brand-col", default="Brand", help="Brand column")
    parser.add_argument("--desc-col", default="Body (HTML)", help="Description column")
    parser.add_argument("--cat-col", default="", help="Category column (optional)")
    parser.add_argument("--sku-col", default="", help="SKU/Barcode column (optional)")
    parser.add_argument("--model-col", default="", help="Model/Code column (optional)")
    
    # API settings
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), 
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--temp", type=float, default=None, 
                        help="Sampling temperature (optional)")
    parser.add_argument("--timeout", type=float, default=60.0, 
                        help="HTTP timeout seconds per request")
    
    # Processing settings
    parser.add_argument("--workers", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--checkpoint-every", type=int, default=100, 
                        help="Save checkpoint every N rows")
    parser.add_argument("--sample", type=int, default=0, 
                        help="Process only first N rows (0 for all)")
    parser.add_argument("--desc-cap", type=int, default=900, 
                        help="Max characters of description sent to model")
    parser.add_argument("--no-schema", action="store_true", 
                        help="Disable strict JSON schema validation")
    
    args = parser.parse_args()
    
    # Validate API key
    if not (args.api_key or os.getenv("OPENAI_API_KEY")):
        raise SystemExit("Error: Set OPENAI_API_KEY environment variable or pass --api-key")
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    # Parse paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise SystemExit(f"Error: Input file not found: {input_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optional columns
    cat_col = args.cat_col if args.cat_col else None
    sku_col = args.sku_col if args.sku_col else None
    model_col = args.model_col if args.model_col else None
    
    # Show configuration
    logger.info("=" * 60)
    logger.info("Universal Translation and SEO Generator (Generic)")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Context: {args.context}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Columns: name={args.name_col}, brand={args.brand_col}, desc={args.desc_col}")
    if cat_col:
        logger.info(f"Category column: {cat_col}")
    if sku_col:
        logger.info(f"SKU column: {sku_col}")
    if model_col:
        logger.info(f"Model column: {model_col}")
    if args.system_prompt:
        logger.info("Using custom system prompt")
    if args.meta_title_prompt:
        logger.info("Using custom meta title prompt")
    if args.meta_desc_prompt:
        logger.info("Using custom meta description prompt")
    if args.product_title_prompt:
        logger.info("Using custom product title prompt")
    logger.info("=" * 60)
    
    # Run processing
    try:
        stats = asyncio.run(process_excel_file(
            input_path=input_path,
            output_path=output_path,
            name_col=args.name_col,
            brand_col=args.brand_col,
            desc_col=args.desc_col,
            cat_col=cat_col,
            sku_col=sku_col,
            model_col=model_col,
            api_key=api_key,
            model=args.model,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
            sample=args.sample,
            desc_cap=args.desc_cap,
            temperature=args.temp,
            no_schema=args.no_schema,
            timeout=args.timeout,
            product_context=args.context,
            custom_system_prompt=args.system_prompt,
            custom_meta_title_prompt=args.meta_title_prompt,
            custom_meta_desc_prompt=args.meta_desc_prompt,
            custom_product_title_prompt=args.product_title_prompt
        ))
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE!")
        logger.info(f"Statistics: {stats}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
