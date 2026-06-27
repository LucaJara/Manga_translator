"""
Manga/Comic Translator - Pipeline 100% local (sem Ollama)

Reconhecimento:
    - Detecção de balões: comic-text-detector (ONNX, independente de idioma)
    - OCR japonês: manga-ocr
    - OCR demais idiomas: EasyOCR
Tradução:
    - Google Translate (balão a balão)

Requisitos:
    pip install pillow keyboard
    pip install onnxruntime opencv-python manga-ocr easyocr

Uso:
    python manga_translator.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import os
import json
import re
import collections
import urllib.request
import urllib.parse
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

try:
    from PIL import ImageGrab, Image, ImageFilter
except ImportError:
    print("Instale: pip install pillow"); sys.exit(1)

try:
    import keyboard
except ImportError:
    print("Instale: pip install keyboard"); sys.exit(1)


# --- Armazenamento dos modelos: tudo em ./models, ao lado deste script ---
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EASYOCR_DIR = os.path.join(MODELS_DIR, "easyocr")
os.makedirs(EASYOCR_DIR, exist_ok=True)
# manga-ocr / HuggingFace localizam o cache pela variável HF_HOME.
# Definido antes de qualquer import de transformers/manga_ocr.
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")

# --- Glossário persistente de nomes de personagens ---
GLOSSARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossary.json")

def _load_glossary():
    try:
        with open(GLOSSARY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_glossary(glossary):
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


MAX_TRANSLATE_CHARS = 4500

# Modelo do tradutor com contexto (LLM local via transformers; usa a GPU
# automaticamente quando há CUDA). Trocável por "Qwen/Qwen2.5-1.5B-Instruct"
# em máquinas com menos memória.
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Se os modelos HF já estão em cache, força o modo offline. Isso evita as
# checagens de metadados em rede a cada inicialização — checagens que disparam
# o aviso "unauthenticated requests to the HF Hub", facilmente confundido com
# um novo download (na realidade o modelo só é carregado do disco para a RAM).
_HF_HUB_DIR = os.path.join(MODELS_DIR, "huggingface", "hub")
if (os.path.isdir(os.path.join(_HF_HUB_DIR,
                               "models--kha-white--manga-ocr-base"))
    and os.path.isdir(os.path.join(
        _HF_HUB_DIR, "models--" + LLM_MODEL_ID.replace("/", "--")))):
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

BUBBLE_COLORS = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#FF9A3C", "#C77DFF", "#00C9A7", "#FF6FD8",
    "#F9F871", "#A8DADC"
]

SOURCE_LANGUAGES = {
    "日本語 (Japonês)":  "ja",
    "中文 (Chinês)":     "zh",
    "한국어 (Coreano)":  "ko",
    "English":          "en",
    "Português":        "pt",
    "Español":          "es",
    "Français":         "fr",
    "Deutsch":          "de",
    "Italiano":         "it",
    "Tiếng Việt":       "vi",
    "Русский":          "ru",
    "العربية":          "ar",
}

TARGET_LANGUAGES = {
    "Português":  "pt",
    "English":    "en",
    "Español":    "es",
    "Français":   "fr",
    "Italiano":   "it",
    "Deutsch":    "de",
    "中文":        "zh-CN",
    "한국어":      "ko",
    "Русский":    "ru",
    "日本語":      "ja",
}

LANG_DISPLAY = {
    "ja": "Japonês", "zh": "Chinês", "zh-cn": "Chinês",
    "ko": "Coreano", "en": "Inglês", "pt": "Português",
    "es": "Espanhol", "fr": "Francês", "de": "Alemão",
    "it": "Italiano", "ru": "Russo", "ar": "Árabe",
    "vi": "Vietnamita", "auto": "Auto",
}

def validate_and_fix_bubbles(bubbles):
    """
    Pós-processamento das coordenadas retornadas pelo modelo:
    - Remove duplicatas (mesmo texto ou caixas muito sobrepostas)
    - Corrige coordenadas fora dos limites (0-1000)
    - Garante tamanho mínimo de caixa
    - Reordena por posição de leitura (direita→esquerda, cima→baixo para japonês)
    """
    if not bubbles:
        return []

    fixed = []
    for b in bubbles:
        try:
            x = max(0, min(990, int(b.get("x", 0))))
            y = max(0, min(990, int(b.get("y", 0))))
            w = max(20, min(1000 - x, int(b.get("w", 100))))
            h = max(20, min(1000 - y, int(b.get("h", 60))))
            text = str(b.get("text", "")).strip()
            if not text:
                continue
            fixed.append({
                "id":   b.get("id", len(fixed) + 1),
                "type": b.get("type", "speech_bubble"),
                "x": x, "y": y, "w": w, "h": h,
                "text": text
            })
        except Exception:
            continue

    # Helpers de geometria: IoU e containment (fração da menor dentro da maior).
    def box_metrics(a, b):
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        aa = a["w"] * a["h"]
        bb = b["w"] * b["h"]
        union = aa + bb - inter
        iou_v = inter / union if union > 0 else 0
        small = min(aa, bb)
        cont  = inter / small if small > 0 else 0
        return iou_v, cont

    def norm_text(t):
        # texto normalizado para comparação: minúsculas, sem pontuação/espaço
        return re.sub(r'[\s\W_]+', '', t.lower(), flags=re.UNICODE)

    # Dedup combinada — maior caixa primeiro (a "principal" do balão).
    # Descarta um candidato quando ele coincide com algo já mantido por:
    #   (a) sobreposição geométrica forte (IoU > 0.5 ou containment > 0.7) —
    #       cobre o caso de o detector emitir duas caixas para o mesmo balão;
    #   (b) mesmo texto normalizado *e* alguma proximidade geométrica — cobre
    #       variações de OCR sobre o mesmo balão (ex.: "Olá" vs "Olá!").
    # Caixas distantes com o mesmo texto (dois personagens dizendo "Sim!") são
    # preservadas — não há sobreposição entre elas.
    fixed.sort(key=lambda b: -(b["w"] * b["h"]))

    kept = []
    for b in fixed:
        key = norm_text(b["text"])
        drop = False
        for k in kept:
            iou_v, cont_v = box_metrics(b, k)
            if iou_v > 0.5 or cont_v > 0.7:
                drop = True
                break
            if key and key == norm_text(k["text"]) and (iou_v > 0.1
                                                        or cont_v > 0.3):
                drop = True
                break
        if not drop:
            kept.append(b)

    # Reordena por leitura: cima→baixo, dentro da faixa direita→esquerda (JP).
    def reading_order(b):
        row = b["y"] // 150
        col = 1000 - b["x"]
        return (row, col)

    kept.sort(key=reading_order)
    for i, b in enumerate(kept):
        b["id"] = i + 1
    return kept


def is_meaningful_text(t):
    """True se o texto tem ao menos um caractere alfanumérico (Unicode, cobre
    CJK). Filtra lixo de OCR sobre SFX/ruído: '...', '!?', '——' etc."""
    return any(c.isalnum() for c in t)


def get_virtual_screen(parent=None):
    """Retorna (x, y, w, h) do desktop virtual — a área que engloba TODOS os
    monitores. A origem pode ser negativa (monitores à esquerda/acima do
    primário). No Windows usa a API de métricas; senão, cai para o monitor
    primário via tkinter.
    """
    try:
        g = ctypes.windll.user32.GetSystemMetrics
        # SM_X/YVIRTUALSCREEN=76/77, SM_CX/CYVIRTUALSCREEN=78/79
        x, y, w, h = g(76), g(77), g(78), g(79)
        if w > 0 and h > 0:
            return x, y, w, h
    except Exception:
        pass
    if parent is not None:
        return 0, 0, parent.winfo_screenwidth(), parent.winfo_screenheight()
    return 0, 0, 1920, 1080


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_dark(hex_color, factor=0.35):
    r, g, b = hex_to_rgb(hex_color)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def preprocess_capture(pil_img, max_dim=2400):
    """Melhora a imagem capturada para detecção/OCR: upscale + denoising.

    O upscaling 2x (Lanczos) dá mais detalhe de caracteres pequenos ao OCR;
    o filtro mediano remove ruído de scan sem borrar as bordas do traço.
    """
    img = pil_img.convert("RGB")
    scale = 2.0
    longest = max(img.width, img.height)
    if longest * scale > max_dim:
        scale = max_dim / longest
    if scale > 1.0:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS
        )
    return img.filter(ImageFilter.MedianFilter(size=3))


def sample_bubble_colors(pil_img, box):
    """Estima a cor de fundo e a de texto de uma região de balão.

    `box` = (x1, y1, x2, y2) em pixels. A cor mais frequente da região é
    tratada como fundo (o texto ocupa menos área); o texto recebe preto ou
    branco conforme a luminância do fundo. Retorna duas strings '#rrggbb'.
    """
    try:
        crop = pil_img.convert("RGB").crop(box)
        crop.thumbnail((48, 48))
        colors = crop.getcolors(48 * 48)
        if not colors:
            return "#ffffff", "#000000"
        bg = max(colors, key=lambda c: c[0])[1]
    except Exception:
        return "#ffffff", "#000000"
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg  = (0, 0, 0) if lum > 140 else (255, 255, 255)
    return "#%02x%02x%02x" % bg, "#%02x%02x%02x" % fg


CTD_MODEL_FILENAME = "comic-text-detector.onnx"
CTD_MODEL_URL = (
    "https://huggingface.co/mayocream/comic-text-detector-onnx/"
    "resolve/main/comic-text-detector.onnx"
)


class BubbleDetector:
    """Detecção de regiões de texto via comic-text-detector (ONNX).

    Usa a saída 'blk' do modelo (blocos de texto com coordenadas já decodificadas
    em escala 1024) aplicando NMS. Carregamento preguiçoso: o modelo só é
    baixado/carregado no primeiro uso.
    """

    INPUT_SIZE  = 1024
    CONF_THRESH = 0.4
    NMS_THRESH  = 0.35

    def __init__(self):
        self._session    = None
        self._input_name = None

    def _model_path(self):
        return os.path.join(MODELS_DIR, CTD_MODEL_FILENAME)

    def _ensure_loaded(self, status_cb=None):
        if self._session is not None:
            return
        import onnxruntime as ort
        path = self._model_path()
        if not os.path.exists(path):
            if status_cb:
                status_cb("Baixando modelo de detecção (~95 MB)...")
            urllib.request.urlretrieve(CTD_MODEL_URL, path)
        if status_cb:
            status_cb("Carregando detector de balões...")
        self._session    = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, pil_img, status_cb=None):
        """Retorna lista de caixas (x, y, w, h) em pixels da imagem recebida.

        Lança exceção se o modelo/dependências não estiverem disponíveis.
        """
        import cv2
        import numpy as np

        self._ensure_loaded(status_cb)

        rgb = np.asarray(pil_img)
        oh, ow = rgb.shape[:2]
        size  = self.INPUT_SIZE
        ratio = min(size / oh, size / ow)
        nh, nw = int(round(oh * ratio)), int(round(ow * ratio))

        # letterbox: redimensiona mantendo proporção, preenche o resto
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized

        inp = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        blk = self._session.run(None, {self._input_name: inp})[0][0]  # (N, 7)

        # blk: [cx, cy, w, h, objectness, classe0, classe1]
        scores = blk[:, 4] * blk[:, 5:].max(axis=1)
        keep   = scores > self.CONF_THRESH
        blk, scores = blk[keep], scores[keep]
        if len(blk) == 0:
            return []

        rects = [
            [float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)]
            for cx, cy, bw, bh in blk[:, :4]
        ]
        idxs = cv2.dnn.NMSBoxes(
            rects, scores.tolist(), self.CONF_THRESH, self.NMS_THRESH
        )
        if len(idxs) == 0:
            return []

        boxes = []
        for i in np.array(idxs).flatten():
            rx, ry, rw, rh = rects[int(i)]
            # desfaz o letterbox: escala 1024 -> pixels da imagem original
            x = max(0, int(rx / ratio))
            y = max(0, int(ry / ratio))
            w = min(ow - x, int(rw / ratio))
            h = min(oh - y, int(rh / ratio))
            if w < 10 or h < 10:
                continue
            # margem pequena para o OCR não cortar o traço
            pad = int(0.04 * max(w, h))
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(ow - x, w + 2 * pad)
            h = min(oh - y, h + 2 * pad)
            boxes.append((x, y, w, h))
        return self._merge_boxes(boxes)

    @staticmethod
    def _merge_boxes(boxes, thresh=0.30):
        """Funde caixas que pertencem ao mesmo balão (merge transitivo).

        O detector às vezes divide um balão em 2+ caixas (colunas de texto
        vertical). Fundir antes do OCR entrega o balão inteiro num único crop
        — sem perder texto na dedup posterior. Critério conservador:
        interseção / área da menor caixa > `thresh`.
        """
        n = len(boxes)
        if n < 2:
            return boxes

        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def overlap_frac(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            ih = max(0, min(ay + ah, by + bh) - max(ay, by))
            small = min(aw * ah, bw * bh)
            return (iw * ih) / small if small > 0 else 0.0

        for i in range(n):
            for j in range(i + 1, n):
                if overlap_frac(boxes[i], boxes[j]) > thresh:
                    parent[find(i)] = find(j)

        groups = {}
        for i, box in enumerate(boxes):
            groups.setdefault(find(i), []).append(box)

        merged = []
        for grp in groups.values():
            x1 = min(b[0] for b in grp)
            y1 = min(b[1] for b in grp)
            x2 = max(b[0] + b[2] for b in grp)
            y2 = max(b[1] + b[3] for b in grp)
            merged.append((x1, y1, x2 - x1, y2 - y1))
        return merged


class MangaOCREngine:
    """OCR especializado em japonês via manga-ocr. Carregamento preguiçoso."""

    def __init__(self):
        self._mocr = None

    def _ensure_loaded(self, status_cb=None):
        if self._mocr is not None:
            return
        if status_cb:
            status_cb("Carregando manga-ocr (~400 MB no 1º uso)...")
        from manga_ocr import MangaOcr
        self._mocr = MangaOcr()

    def read(self, pil_crop, status_cb=None):
        self._ensure_loaded(status_cb)
        return str(self._mocr(pil_crop)).strip()


class EasyOCREngine:
    """OCR multilíngue via EasyOCR para idiomas não-japoneses.

    Mantém um `easyocr.Reader` em cache por idioma; cada um é carregado de forma
    preguiçosa no primeiro uso (baixa o modelo do idioma na 1ª vez).
    """

    # Códigos dos idiomas de origem -> códigos do EasyOCR
    LANG_MAP = {
        "zh": "ch_sim", "ko": "ko", "en": "en", "pt": "pt",
        "es": "es", "fr": "fr", "de": "de", "it": "it",
        "ru": "ru", "ar": "ar", "vi": "vi",
    }
    # Idiomas latinos cujo Reader se beneficia de combinar com inglês —
    # vietnamita usa o alfabeto latino com muitos diacríticos e a "âncora"
    # de inglês ajuda o modelo a preservar os tons/acentos.
    _PAIR_WITH_EN = {"vi"}

    def __init__(self):
        self._readers = {}   # código easyocr -> Reader

    def _reader_langs(self, key):
        return [key, "en"] if key in self._PAIR_WITH_EN else [key]

    def _get_reader(self, lang_code, status_cb=None):
        key = self.LANG_MAP.get(lang_code, "en")
        if key not in self._readers:
            if status_cb:
                status_cb(f"Carregando EasyOCR ({key})...")
            import easyocr, torch
            self._readers[key] = easyocr.Reader(
                self._reader_langs(key),
                gpu=torch.cuda.is_available(),
                model_storage_directory=EASYOCR_DIR,
                download_enabled=True,
            )
        return self._readers[key]

    def read(self, pil_crop, lang_code, status_cb=None):
        import numpy as np
        reader = self._get_reader(lang_code, status_cb)
        arr = np.asarray(pil_crop)
        # mag_ratio=2.0 dobra a resolução interna — essencial para preservar
        # diacríticos pequenos (acentos/tons). decoder='beamsearch' faz
        # decodificação com busca em feixe (mais acurado que o 'greedy').
        lines = reader.readtext(
            arr, detail=0, paragraph=True,
            mag_ratio=2.0, decoder="beamsearch",
        )
        return " ".join(str(t) for t in lines).strip()


class LLMTranslatorEngine:
    """Tradutor com contexto de diálogo via LLM local (transformers, em CPU).

    Traduz todos os balões de uma página numa única chamada e recebe o diálogo
    das páginas anteriores como contexto. Carregamento preguiçoso; o modelo é
    cacheado em models/huggingface/ (via HF_HOME).
    """

    def __init__(self):
        self._tok   = None
        self._model = None
        self.device = "cpu"

    def _ensure_loaded(self, status_cb=None):
        if self._model is not None:
            return
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        onde = "GPU" if self.device == "cuda" else "CPU"
        if status_cb:
            status_cb(f"Carregando modelo de tradução (LLM, {onde})...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, dtype="auto"
            )
        except TypeError:                       # transformers mais antigo
            self._model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, torch_dtype="auto"
            )
        self._model.to(self.device)
        self._model.eval()

    # Nomes em inglês dos idiomas de origem para o prompt do LLM.
    _SRC_LANG_EN = {
        "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
        "en": "English",  "pt": "Portuguese", "es": "Spanish",
        "fr": "French",   "de": "German",     "it": "Italian",
        "vi": "Vietnamese", "ru": "Russian",  "ar": "Arabic",
    }

    def translate(self, texts, tgt_name, src_code=None, history=None, glossary=None, status_cb=None):
        """Traduz a lista de falas para `tgt_name` numa só chamada, usando o
        diálogo anterior (`history`: sequência de pares (origem, tradução))
        como contexto. `src_code` (ex.: 'ja', 'vi') é incluído no prompt para
        melhorar coerência (honoríficos, pronomes omitidos, etc.).

        Retorna a lista de traduções. Lança exceção se a saída for inválida.
        """
        import torch
        self._ensure_loaded(status_cb)
        if status_cb:
            onde = "GPU" if self.device == "cuda" else "CPU"
            status_cb(f"Traduzindo com contexto (LLM, {onde})...")

        gloss_block = ""
        if glossary:
            entries = "\n".join(f"- {src} → {tgt}" for src, tgt in glossary.items())
            gloss_block = (
                "Character/name glossary — always use these exact translations, "
                "never translate or romanize them differently:\n"
                f"{entries}\n\n"
            )

        ctx = ""
        if history:
            pairs = "\n".join(f"- {s}  =>  {t}" for s, t in history)
            ctx = (
                "Earlier dialogue from the same story, already translated "
                "(reference only — do NOT translate it again):\n"
                f"{pairs}\n\n"
            )
        numbered = "\n".join(
            f"{i+1}. {t.replace(chr(10), ' ').strip()}"
            for i, t in enumerate(texts)
        )
        src_label = self._SRC_LANG_EN.get(src_code, "") if src_code else ""
        from_clause = f" from {src_label}" if src_label else ""
        user_msg = (
            f"You are a professional manga/comic translator. Translate the "
            f"numbered lines below{from_clause} into {tgt_name}.\n\n"
            f"The lines are dialogue read in order — use the conversation "
            f"context to keep tone, pronouns, honorifics and each character's "
            f"voice consistent.\n\n"
            f"Translation guidelines:\n"
            f"- Translate naturally and idiomatically, never word-for-word; "
            f"rephrase whenever it makes the line sound more native.\n"
            f"- Keep pronouns consistent with the conversation context: "
            f"Asian languages often omit the subject, so infer it from the "
            f"surrounding dialogue — do not flip \"I\"↔\"you\" between "
            f"consecutive lines unless the speaker clearly changes. When "
            f"singular vs plural \"you\" is ambiguous in English, prefer "
            f"singular.\n"
            f"- Use the conversational register a native {tgt_name} speaker "
            f"would use in a comic — short, punchy lines, informal pronouns "
            f"and native interjections. Avoid formal or literary phrasing.\n"
            f"- Preserve emotional intensity: exclamations stay exclamations, "
            f"hesitations and ellipses (…/—) stay; render interjections with "
            f"natural equivalents in {tgt_name}.\n"
            f"- Keep proper names, place names and onomatopoeia (sound effects) "
            f"unchanged.\n"
            f"- Keep each line concise — manga balloons have limited space.\n\n"
            f"{gloss_block}"
            f"{ctx}"
            f"Lines to translate:\n{numbered}\n\n"
            f"Reply ONLY with the numbered translations, one per line, with the "
            f"same numbers and nothing else."
        )
        messages = [
            {"role": "system",
             "content": ("You translate manga/comic dialogue into natural, "
                         "idiomatic target-language text. You preserve "
                         "emotional tone and you never translate word-for-word.")},
            {"role": "user", "content": user_msg},
        ]
        max_new = min(1024, 256 + len(numbered))

        # 1ª tentativa: beam search (determinístico). Se a saída vier
        # malformada, 2ª tentativa com amostragem — gera texto diferente,
        # dando uma chance real antes do fallback (Google, sem contexto).
        raw = self._generate(messages, max_new, sample=False)
        try:
            return self._parse(raw, len(texts))
        except ValueError:
            raw = self._generate(messages, max_new, sample=True)
            return self._parse(raw, len(texts))

    def _generate(self, messages, max_new_tokens, sample):
        import torch
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        if sample:
            gen_kwargs = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            gen_kwargs = dict(do_sample=False, num_beams=2,
                              early_stopping=True)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.1,
                pad_token_id=self._tok.eos_token_id,
                **gen_kwargs,
            )
        gen = out[0][inputs["input_ids"].shape[1]:].cpu()
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    @staticmethod
    def _parse(raw, n):
        parsed = {}
        for line in raw.splitlines():
            m = re.match(r'\s*(\d+)\s*[.):\-]\s*(.+)', line)
            if m:
                t = m.group(2).strip()
                # remove aspas envolventes que o modelo às vezes adiciona
                if len(t) > 1 and t[0] in "\"'“‘" and t[-1] in "\"'”’":
                    t = t[1:-1].strip()
                parsed[int(m.group(1))] = t
        result = [parsed.get(i + 1) for i in range(n)]
        if any(not v for v in result):
            raise ValueError("tradução do LLM incompleta ou malformada")
        return result


class MarianTranslatorEngine:
    """Tradução local e offline via Helsinki-NLP/opus-mt (MarianMT).

    Modelos pequenos (~300 MB cada), carregados por par de idiomas sob demanda.
    Muito mais rápido que o LLM Qwen e não exige internet após o download.
    Sem contexto de diálogo — use o LLM quando isso importa.

    Suporta dois mecanismos além da tradução direta:
    - _LANG_PREFIX: modelos multi-alvo (ex.: opus-mt-en-ROMANCE) exigem um
      prefixo de idioma no texto de entrada, ex.: ">>pt<< ".
    - _CHAIN: pares sem modelo direto são encadeados em dois passos,
      ex.: vi→pt = vi→en seguido de en→pt.

    Prioridade de uso: LLM (contexto) → Marian (offline) → Google (online).
    """

    # Pares (src_code, tgt_code) -> model_id HuggingFace com modelo direto.
    _MODEL_MAP = {
        ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
        ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
        ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
        ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
        ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
        # opus-mt-en-ROMANCE cobre en→{pt,es,fr,it,...} com prefixo de idioma.
        ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",
    }

    # Prefixo obrigatório para modelos multi-alvo (ex.: ROMANCE).
    # O prefixo é adicionado a cada frase antes da tokenização e removido
    # automaticamente na decodificação pelo MarianTokenizer.
    _LANG_PREFIX = {
        ("en", "pt"): ">>pt<< ",
    }

    # Pares sem modelo direto: encadeamento em dois passos (A→B→C).
    # Cada par intermediário deve existir em _MODEL_MAP ou _CHAIN.
    _CHAIN = {
        ("ja", "pt"): [("ja", "en"), ("en", "pt")],
    }

    def __init__(self):
        self._models = {}   # (src, tgt) -> (tokenizer, model)

    def supports(self, src_code, tgt_code):
        """True se o par é suportado, diretamente ou via encadeamento."""
        return ((src_code, tgt_code) in self._MODEL_MAP
                or (src_code, tgt_code) in self._CHAIN)

    def _ensure_loaded(self, src_code, tgt_code, status_cb=None):
        key = (src_code, tgt_code)
        if key in self._models:
            return self._models[key]

        model_id  = self._MODEL_MAP[key]
        hub_dir   = os.path.join(MODELS_DIR, "huggingface", "hub")
        is_cached = os.path.isdir(
            os.path.join(hub_dir, "models--" + model_id.replace("/", "--"))
        )

        # Se o modelo ainda não está em disco, suspende o modo offline
        # temporariamente para permitir o download — sem afetar os outros
        # modelos que já foram carregados.
        saved = {k: os.environ.pop(k, None)
                 for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            if not is_cached and status_cb:
                status_cb(
                    f"Baixando modelo Marian {src_code}→{tgt_code} (~300 MB)..."
                )
            if status_cb:
                status_cb(f"Carregando tradutor local ({src_code}→{tgt_code})...")
            from transformers import MarianMTModel, MarianTokenizer
            tok = MarianTokenizer.from_pretrained(model_id)
            mdl = MarianMTModel.from_pretrained(model_id)
            mdl.eval()
            self._models[key] = (tok, mdl)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        return self._models[key]

    def _translate_direct(self, texts, src_code, tgt_code, status_cb=None):
        """Tradução num único passo (um modelo). Aplica prefixo de idioma
        quando o modelo multi-alvo exige (ex.: opus-mt-en-ROMANCE com >>pt<<).
        """
        import torch
        tok, mdl = self._ensure_loaded(src_code, tgt_code, status_cb)
        if status_cb:
            status_cb(f"Traduzindo ({src_code}→{tgt_code}, Marian)...")
        prefix = self._LANG_PREFIX.get((src_code, tgt_code), "")
        tagged = [prefix + t for t in texts] if prefix else texts
        inputs = tok(
            tagged, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        with torch.no_grad():
            out = mdl.generate(**inputs, num_beams=4, max_length=512)
        return [tok.decode(t, skip_special_tokens=True) for t in out]

    def translate(self, texts, src_code, tgt_code, status_cb=None):
        """Traduz a lista de textos. Suporta pares encadeados (dois passos)
        e pares directos com ou sem prefixo de idioma.
        Retorna lista de strings traduzidas.
        """
        chain = self._CHAIN.get((src_code, tgt_code))
        if chain:
            result = texts
            for s, t in chain:
                result = self._translate_direct(result, s, t, status_cb)
            return result
        return self._translate_direct(texts, src_code, tgt_code, status_cb)


# Ícones por tipo de balão
BUBBLE_TYPE_ICONS = {
    "speech_bubble":  "💬",
    "spike_bubble":   "❗",
    "thought_bubble": "💭",
    "narration_box":  "📖",
    "title_box":      "📌",
    "sfx":            "🔊",
    "caption":        "📍",
}


class BubbleOverlay:
    """Janela transparente com overlay aprimorado de balões."""

    def __init__(self):
        self.win = tk.Toplevel()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", "#000001")
        self.win.configure(bg="#000001")
        self.win.withdraw()

        self.canvas = tk.Canvas(self.win, bg="#000001", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.region    = None
        self.bubbles   = []
        self.active_id = None
        self._pulse_job   = None
        self._pulse_state = True

    def set_region(self, region):
        self.region = region
        x, y, x2, y2 = region
        self.win.geometry(f"{x2-x}x{y2-y}+{x}+{y}")
        self.win.deiconify()

    def draw_bubbles(self, bubbles, active_id=None):
        self._stop_pulse()
        self.bubbles   = bubbles
        self.active_id = active_id
        self._render(pulse_on=True)
        if self.region:
            self.win.deiconify()
        if active_id is not None:
            self._start_pulse()

    def draw_replacements(self, bubbles):
        """Modo substituição: cobre o texto original e desenha a tradução
        dentro de cada balão."""
        self._stop_pulse()
        self.bubbles   = bubbles
        self.active_id = None
        self._render_replacements()
        if self.region:
            self.win.deiconify()

    def _render_replacements(self):
        self.canvas.delete("all")
        if not self.region or not self.bubbles:
            return
        rx, ry, rx2, ry2 = self.region
        rw, rh = rx2 - rx, ry2 - ry
        for b in self.bubbles:
            x1 = int(b["x"] / 1000 * rw)
            y1 = int(b["y"] / 1000 * rh)
            x2 = int((b["x"] + b["w"]) / 1000 * rw)
            y2 = int((b["y"] + b["h"]) / 1000 * rh)
            bg = b.get("bg", "#ffffff")
            fg = b.get("fg", "#000000")
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            text = b.get("text", "")
            if not text:
                # Sem texto traduzido: cobre a caixa toda (caso raro).
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg, outline=bg)
                continue

            # Área disponível = caixa detectada com margem interna de 8%.
            box_w = max(8, (x2 - x1) - 2 * max(2, int((x2 - x1) * 0.08)))
            box_h = max(8, (y2 - y1) - 2 * max(2, int((y2 - y1) * 0.08)))
            size, fitted = self._fit_text(text, box_w, box_h)
            tw, th = self._measure(fitted, size, box_w)

            # Cobertura = o BLOCO DE TEXTO traduzido + margem, centrado na caixa
            # e preso aos limites dela. Assim a área branca acompanha o tamanho
            # da tradução em vez de preencher toda a caixa do detector — que às
            # vezes engloba duas caixas em escada + arte (o detector retorna um
            # único bounding box para o par). Nunca excede a caixa detectada.
            pad = max(4, size // 2)
            cw = min(x2 - x1, tw + 2 * pad)
            ch = min(y2 - y1, th + 2 * pad)
            bx1 = max(x1, cx - cw // 2)
            by1 = max(y1, cy - ch // 2)
            bx2 = min(x2, bx1 + cw)
            by2 = min(y2, by1 + ch)
            self.canvas.create_rectangle(bx1, by1, bx2, by2, fill=bg, outline=bg)
            self.canvas.create_text(
                cx, cy,
                text=fitted, fill=fg, font=("Arial", size),
                width=box_w, justify="center"
            )

    def _measure(self, text, size, max_w):
        """Largura e altura do texto quebrado em `max_w` na fonte Arial `size`."""
        item = self.canvas.create_text(
            -3000, -3000, text=text, font=("Arial", size),
            width=max_w, justify="center", anchor="center"
        )
        bx = self.canvas.bbox(item)
        self.canvas.delete(item)
        if not bx:
            return 0, 0
        return bx[2] - bx[0], bx[3] - bx[1]

    def _fit_text(self, text, max_w, max_h, size_max=40, size_min=7):
        """Retorna (tamanho_da_fonte, texto) que CABE na caixa (max_w × max_h).

        Busca binária pela maior fonte cujo texto quebrado cabe (~6 medições
        em vez de até 34). Se nem no menor tamanho couber (balão minúsculo /
        texto muito longo), trunca o texto com '…' — garantindo que a
        renderização nunca exceda a área.
        """
        lo, hi, best_size = size_min, size_max, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            w, h = self._measure(text, mid, max_w)
            if w <= max_w and h <= max_h:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best_size >= size_min:
            return best_size, text

        # não coube nem no mínimo: trunca com reticências (busca binária)
        size = size_min
        lo, hi, best = 0, len(text), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[:mid].rstrip() + "…"
            w, h = self._measure(cand, size, max_w)
            if w <= max_w and h <= max_h:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return size, (best or "…")

    def _render(self, pulse_on=True):
        self.canvas.delete("all")
        if not self.region or not self.bubbles:
            return

        rx, ry, rx2, ry2 = self.region
        rw, rh = rx2 - rx, ry2 - ry

        for bubble in self.bubbles:
            bid       = bubble["id"]
            btype     = bubble.get("type", "speech_bubble")
            color     = BUBBLE_COLORS[(bid - 1) % len(BUBBLE_COLORS)]
            dark      = make_dark(color)
            is_active = (bid == self.active_id)
            bw        = 3 if is_active and pulse_on else 2

            x1 = int(bubble["x"] / 1000 * rw)
            y1 = int(bubble["y"] / 1000 * rh)
            x2 = int((bubble["x"] + bubble["w"]) / 1000 * rw)
            y2 = int((bubble["y"] + bubble["h"]) / 1000 * rh)

            # Estilo de borda por tipo
            dash = None
            if btype in ("narration_box", "title_box"):
                dash = (6, 3)       # tracejado para caixas de narração
            elif btype == "caption":
                dash = (2, 4)       # pontilhado para legendas flutuantes
            elif btype == "sfx":
                dash = (8, 2)       # traço longo para SFX

            # Sombra externa
            self.canvas.create_rectangle(
                x1+2, y1+2, x2+2, y2+2,
                outline=dark, width=bw+1, fill="", dash=dash or ()
            )
            # Borda principal
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=bw, fill="", dash=dash or ()
            )

            # Anel de pulso quando ativo
            if is_active and pulse_on:
                self.canvas.create_rectangle(
                    x1-4, y1-4, x2+4, y2+4,
                    outline=color, width=1, fill=""
                )

            # Badge circular com número
            badge_r  = 12
            badge_cx = x1 + badge_r + 3
            badge_cy = y1 + badge_r + 3

            self.canvas.create_oval(
                badge_cx-badge_r+1, badge_cy-badge_r+1,
                badge_cx+badge_r+1, badge_cy+badge_r+1,
                fill=dark, outline=""
            )
            self.canvas.create_oval(
                badge_cx-badge_r, badge_cy-badge_r,
                badge_cx+badge_r, badge_cy+badge_r,
                fill=color, outline="white", width=1
            )
            self.canvas.create_text(
                badge_cx, badge_cy,
                text=str(bid), fill="black",
                font=("Arial", 8, "bold")
            )

            # Tag de tipo (pequena, ao lado do badge)
            type_label = btype.replace("_", " ").upper()[:10]
            lx = badge_cx + badge_r + 4
            ly = badge_cy - 6
            tw = len(type_label) * 5 + 4
            self.canvas.create_rectangle(
                lx-1, ly, lx+tw, ly+12,
                fill=dark, outline=""
            )
            self.canvas.create_text(
                lx+1, ly+6,
                text=type_label, fill=color,
                font=("Arial", 7), anchor="w"
            )

            # Preview de texto ao ativar
            if is_active and pulse_on:
                preview = bubble.get("text", "")[:25].replace("\n", " ")
                if preview:
                    px = x1 + 2
                    py = y2 - 14
                    pw = len(preview) * 6 + 6
                    self.canvas.create_rectangle(
                        px-1, py-1, px+pw, py+13,
                        fill=dark, outline=""
                    )
                    self.canvas.create_text(
                        px+2, py+6,
                        text=preview, fill=color,
                        font=("Arial", 8), anchor="w"
                    )

    def _start_pulse(self):
        self._pulse_state = True
        self._pulse()

    def _pulse(self):
        if self.active_id is None:
            return
        self._pulse_state = not self._pulse_state
        self._render(pulse_on=self._pulse_state)
        self._pulse_job = self.win.after(500, self._pulse)

    def _stop_pulse(self):
        if self._pulse_job:
            try:
                self.win.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None

    def set_active(self, bubble_id):
        self.active_id = bubble_id
        self._stop_pulse()
        self._render(pulse_on=True)
        self._start_pulse()

    def clear_active(self):
        self.active_id = None
        self._stop_pulse()
        self._render(pulse_on=False)

    def hide(self):
        self._stop_pulse()
        self.canvas.delete("all")
        self.win.withdraw()

    def show(self):
        if self.region:
            self.win.deiconify()


class RegionSelector:
    """Seleção de área cobrindo TODOS os monitores (desktop virtual).

    O overlay é posicionado na origem virtual (pode ser negativa) e dimensionado
    para a extensão total. As coordenadas dos eventos são relativas ao canto do
    overlay; somando o offset virtual obtemos coordenadas absolutas de tela, que
    é o que `ImageGrab.grab(all_screens=True)` espera.
    """

    def __init__(self, parent):
        self.region = None
        self.start_x = self.start_y = 0
        self.rect = None
        self.parent = parent

        self.vx, self.vy, vw, vh = get_virtual_screen(parent)

        self.top = tk.Toplevel(parent)
        # geometria explícita cobrindo o desktop virtual; offsets negativos
        # ficam no formato "+-1080" que o Tk aceita. NÃO usar -fullscreen,
        # que prende a janela a um único monitor.
        self.top.geometry(f"{vw}x{vh}+{self.vx}+{self.vy}")
        self.top.attributes("-alpha", 0.3)
        self.top.attributes("-topmost", True)
        self.top.configure(bg="black")
        self.top.overrideredirect(True)

        self.canvas = tk.Canvas(
            self.top, cursor="cross", bg="black", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Texto de instrução centralizado no monitor PRIMÁRIO (origem absoluta
        # 0,0). Em coordenadas do canvas isso fica em (-vx, -vy).
        try:
            g = ctypes.windll.user32.GetSystemMetrics
            pw, ph = g(0), g(1)
        except Exception:
            pw, ph = vw, vh
        self.canvas.create_text(
            -self.vx + pw // 2, -self.vy + 40,
            text="Clique e arraste para selecionar a área (qualquer monitor). "
                 "ESC para cancelar.",
            fill="white", font=("Arial", 14)
        )
        self.canvas.bind("<ButtonPress-1>",   self.on_press)
        self.canvas.bind("<B1-Motion>",        self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.top.bind("<Escape>", lambda e: self.top.destroy())

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)

    def on_drag(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="#534AB7", width=2, fill="#534AB7", stipple="gray25"
        )

    def on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        if (x2-x1) > 20 and (y2-y1) > 20:
            # converte coords do overlay -> coords absolutas de tela
            self.region = (x1 + self.vx, y1 + self.vy,
                           x2 + self.vx, y2 + self.vy)
        self.top.destroy()

    def select(self):
        self.parent.wait_window(self.top)
        return self.region


class OverlayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Universal Comic Translator")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.95)
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("500x600+20+20")
        self.root.resizable(True, True)

        self.bubble_overlay  = BubbleOverlay()
        self.is_running      = False
        self.region          = None
        self.detector        = BubbleDetector()
        self.ocr_engine      = MangaOCREngine()
        self.easyocr_engine  = EasyOCREngine()
        self.llm_translator    = LLMTranslatorEngine()
        self.marian_translator = MarianTranslatorEngine()
        self.dialog_history    = collections.deque(maxlen=24)
        self.glossary          = _load_glossary()
        self._busy_lock      = threading.Lock()

        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#16213e", pady=8)
        header.pack(fill=tk.X)
        tk.Label(header, text="Universal Translator",
                 bg="#16213e", fg="#CECBF6",
                 font=("Arial", 13, "bold")).pack(side=tk.LEFT, padx=12)
        self.engine_lbl = tk.Label(header, text="Motor: —",
                                    bg="#16213e", fg="#9FE1CB",
                                    font=("Arial", 9))
        self.engine_lbl.pack(side=tk.LEFT)
        self.status_lbl = tk.Label(header, text="Parado",
                                    bg="#16213e", fg="#888780",
                                    font=("Arial", 10))
        self.status_lbl.pack(side=tk.RIGHT, padx=12)

        # Botões
        btn_frame = tk.Frame(self.root, bg="#1a1a2e", pady=6)
        btn_frame.pack(fill=tk.X, padx=10)
        self.btn_select = tk.Button(btn_frame, text="Selecionar região",
                                     command=self.select_region,
                                     bg="#534AB7", fg="white", relief=tk.FLAT,
                                     padx=10, pady=5, font=("Arial", 10), cursor="hand2")
        self.btn_select.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_toggle = tk.Button(btn_frame, text="Iniciar",
                                     command=self.toggle_translation,
                                     bg="#3B6D11", fg="white", relief=tk.FLAT,
                                     padx=10, pady=5, font=("Arial", 10), cursor="hand2",
                                     state=tk.DISABLED)
        self.btn_toggle.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_once = tk.Button(btn_frame, text="Traduzir 1x",
                                   command=self.translate_once,
                                   bg="#085041", fg="white", relief=tk.FLAT,
                                   padx=10, pady=5, font=("Arial", 10), cursor="hand2",
                                   state=tk.DISABLED)
        self.btn_once.pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Glossário",
                  command=self._open_glossary_editor,
                  bg="#5A4A1E", fg="white", relief=tk.FLAT,
                  padx=10, pady=5, font=("Arial", 10), cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(6, 0))

        tk.Frame(self.root, bg="#26215C", height=1).pack(fill=tk.X, padx=10, pady=(6,2))

        # Idiomas
        lang_frame = tk.Frame(self.root, bg="#1a1a2e", pady=4)
        lang_frame.pack(fill=tk.X, padx=10)
        tk.Label(lang_frame, text="Ler idioma:",
                 bg="#1a1a2e", fg="#B4B2A9", font=("Arial", 10)
                 ).grid(row=0, column=0, sticky="w", padx=(0,4))
        self.source_lang_var = tk.StringVar(value="日本語 (Japonês)")
        ttk.Combobox(lang_frame, textvariable=self.source_lang_var,
                     values=list(SOURCE_LANGUAGES.keys()),
                     state="readonly", width=18, font=("Arial", 9)
                     ).grid(row=0, column=1, sticky="w", padx=(0,16))
        tk.Label(lang_frame, text="Traduzir para:",
                 bg="#1a1a2e", fg="#B4B2A9", font=("Arial", 10)
                 ).grid(row=0, column=2, sticky="w", padx=(0,4))
        self.target_lang_var = tk.StringVar(value="Português")
        ttk.Combobox(lang_frame, textvariable=self.target_lang_var,
                     values=list(TARGET_LANGUAGES.keys()),
                     state="readonly", width=12, font=("Arial", 9)
                     ).grid(row=0, column=3, sticky="w")

        # Controles
        ctrl_frame = tk.Frame(self.root, bg="#1a1a2e", pady=2)
        ctrl_frame.pack(fill=tk.X, padx=10)
        tk.Label(ctrl_frame, text="Intervalo (s):",
                 bg="#1a1a2e", fg="#B4B2A9", font=("Arial", 10)).pack(side=tk.LEFT)
        self.interval_var = tk.IntVar(value=5)
        tk.Spinbox(ctrl_frame, from_=2, to=60, textvariable=self.interval_var,
                   width=3, bg="#16213e", fg="white", relief=tk.FLAT,
                   font=("Arial", 10), buttonbackground="#16213e"
                   ).pack(side=tk.LEFT, padx=(4,16))
        tk.Label(ctrl_frame, text="Detectado:",
                 bg="#1a1a2e", fg="#B4B2A9", font=("Arial", 10)).pack(side=tk.LEFT)
        self.detected_lbl = tk.Label(ctrl_frame, text="—",
                                      bg="#1a1a2e", fg="#9FE1CB",
                                      font=("Arial", 10, "bold"))
        self.detected_lbl.pack(side=tk.LEFT, padx=(4,16))
        tk.Label(ctrl_frame, text="F9/F10",
                 bg="#1a1a2e", fg="#5F5E5A", font=("Arial", 9)).pack(side=tk.LEFT)

        # Checkbox overlay
        ov_frame = tk.Frame(self.root, bg="#1a1a2e", pady=2)
        ov_frame.pack(fill=tk.X, padx=10)
        self.show_overlay_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ov_frame, text="Mostrar overlay de balões na tela",
                       variable=self.show_overlay_var,
                       bg="#1a1a2e", fg="#B4B2A9",
                       activebackground="#1a1a2e", activeforeground="#EEEDFE",
                       selectcolor="#16213e", font=("Arial", 10), cursor="hand2",
                       command=self._toggle_overlay_visibility).pack(side=tk.LEFT)

        # Checkboxes de seleção de engine de tradução
        engine_frame = tk.Frame(self.root, bg="#1a1a2e", pady=2)
        engine_frame.pack(fill=tk.X, padx=10)
        self.use_marian_var = tk.BooleanVar(value=True)
        tk.Checkbutton(engine_frame,
                       text="Marian (offline, rápido)",
                       variable=self.use_marian_var,
                       bg="#1a1a2e", fg="#B4B2A9",
                       activebackground="#1a1a2e", activeforeground="#EEEDFE",
                       selectcolor="#16213e", font=("Arial", 10), cursor="hand2"
                       ).pack(side=tk.LEFT)
        self.use_llm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(engine_frame,
                       text="LLM com contexto (mais lento)",
                       variable=self.use_llm_var,
                       bg="#1a1a2e", fg="#B4B2A9",
                       activebackground="#1a1a2e", activeforeground="#EEEDFE",
                       selectcolor="#16213e", font=("Arial", 10), cursor="hand2"
                       ).pack(side=tk.LEFT, padx=(16, 0))

        # Checkbox substituir texto nos balões
        rep_frame = tk.Frame(self.root, bg="#1a1a2e", pady=2)
        rep_frame.pack(fill=tk.X, padx=10)
        self.replace_var = tk.BooleanVar(value=False)
        tk.Checkbutton(rep_frame,
                       text="Substituir texto nos balões (sobrepõe a tradução no lugar do original)",
                       variable=self.replace_var,
                       bg="#1a1a2e", fg="#B4B2A9",
                       activebackground="#1a1a2e", activeforeground="#EEEDFE",
                       selectcolor="#16213e", font=("Arial", 10), cursor="hand2"
                       ).pack(side=tk.LEFT)

        self.region_lbl = tk.Label(self.root, text="Nenhuma região selecionada",
                                    bg="#1a1a2e", fg="#5F5E5A", font=("Arial", 9))
        self.region_lbl.pack(padx=10, anchor=tk.W, pady=(2,0))

        tk.Frame(self.root, bg="#26215C", height=1).pack(fill=tk.X, padx=10, pady=6)

        # Área de texto
        text_frame = tk.Frame(self.root, bg="#1a1a2e")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box = tk.Text(text_frame, bg="#0f0f23", fg="#EEEDFE",
                                 font=("Consolas", 11), relief=tk.FLAT,
                                 wrap=tk.WORD, yscrollcommand=scrollbar.set,
                                 padx=10, pady=10, state=tk.DISABLED, cursor="arrow")
        self.text_box.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_box.yview)

        self.text_box.tag_config("header",      foreground="#9FE1CB", font=("Consolas", 10, "bold"))
        self.text_box.tag_config("translation", foreground="#EEEDFE", font=("Consolas", 11))
        self.text_box.tag_config("type_label",  foreground="#888780", font=("Consolas", 9))
        self.text_box.tag_config("error",       foreground="#F09595", font=("Consolas", 10))
        self.text_box.tag_config("info",        foreground="#888780", font=("Consolas", 10))

        for i, color in enumerate(BUBBLE_COLORS):
            self.text_box.tag_config(f"bubble_{i+1}", foreground=color,
                                      font=("Consolas", 11, "bold"))

        keyboard.add_hotkey("F9",  self.translate_once)
        keyboard.add_hotkey("F10", self.stop_translation)

    def _toggle_overlay_visibility(self):
        if self.show_overlay_var.get():
            self.bubble_overlay.show()
        else:
            self.bubble_overlay.hide()

    def _open_glossary_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Glossário de nomes")
        win.configure(bg="#1a1a2e")
        win.resizable(False, False)

        tk.Label(win, text="Glossário de personagens / nomes",
                 bg="#1a1a2e", fg="#CECBF6", font=("Arial", 11, "bold")
                 ).pack(padx=16, pady=(12, 4))
        tk.Label(win, text="As traduções listadas aqui são sempre respeitadas pelo LLM.",
                 bg="#1a1a2e", fg="#888780", font=("Arial", 9)
                 ).pack(padx=16, pady=(0, 8))

        # Lista de entradas
        list_frame = tk.Frame(win, bg="#1a1a2e")
        list_frame.pack(padx=16, fill=tk.BOTH)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                             bg="#16213e", fg="#EEEDFE", selectbackground="#534AB7",
                             font=("Consolas", 10), width=44, height=12, relief=tk.FLAT)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar.config(command=listbox.yview)

        def _refresh():
            listbox.delete(0, tk.END)
            for src, tgt in self.glossary.items():
                listbox.insert(tk.END, f"{src}  →  {tgt}")

        _refresh()

        # Campos de adição
        add_frame = tk.Frame(win, bg="#1a1a2e")
        add_frame.pack(padx=16, pady=(8, 4), fill=tk.X)
        tk.Label(add_frame, text="Original:", bg="#1a1a2e", fg="#B4B2A9",
                 font=("Arial", 9)).grid(row=0, column=0, sticky="w")
        tk.Label(add_frame, text="Tradução:", bg="#1a1a2e", fg="#B4B2A9",
                 font=("Arial", 9)).grid(row=0, column=2, sticky="w", padx=(12, 0))
        var_src = tk.StringVar()
        var_tgt = tk.StringVar()
        tk.Entry(add_frame, textvariable=var_src, width=16,
                 bg="#16213e", fg="#EEEDFE", insertbackground="white",
                 font=("Consolas", 10), relief=tk.FLAT
                 ).grid(row=1, column=0, sticky="w")
        tk.Label(add_frame, text="→", bg="#1a1a2e", fg="#9FE1CB",
                 font=("Arial", 10)).grid(row=1, column=1, padx=4)
        tk.Entry(add_frame, textvariable=var_tgt, width=16,
                 bg="#16213e", fg="#EEEDFE", insertbackground="white",
                 font=("Consolas", 10), relief=tk.FLAT
                 ).grid(row=1, column=2, sticky="w", padx=(12, 0))

        def _add():
            src = var_src.get().strip()
            tgt = var_tgt.get().strip()
            if not src or not tgt:
                return
            self.glossary[src] = tgt
            _save_glossary(self.glossary)
            var_src.set("")
            var_tgt.set("")
            _refresh()

        def _remove():
            sel = listbox.curselection()
            if not sel:
                return
            key = list(self.glossary.keys())[sel[0]]
            del self.glossary[key]
            _save_glossary(self.glossary)
            _refresh()

        def _import_txt():
            from tkinter import filedialog, messagebox
            path = filedialog.askopenfilename(
                parent=win,
                title="Selecionar arquivo de glossário",
                filetypes=[("Arquivo de texto", "*.txt"), ("Todos os arquivos", "*.*")],
            )
            if not path:
                return
            imported = 0
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Suporta: "original → tradução", "original -> tradução",
                        #          "original = tradução", "original\ttradução"
                        for sep in ("→", "->", "=", "\t"):
                            if sep in line:
                                parts = line.split(sep, 1)
                                src, tgt = parts[0].strip(), parts[1].strip()
                                if src and tgt:
                                    self.glossary[src] = tgt
                                    imported += 1
                                break
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível ler o arquivo:\n{e}", parent=win)
                return
            _save_glossary(self.glossary)
            _refresh()
            messagebox.showinfo("Importação concluída",
                                f"{imported} entradas importadas.", parent=win)

        btn_row = tk.Frame(win, bg="#1a1a2e")
        btn_row.pack(padx=16, pady=(4, 12), fill=tk.X)
        tk.Button(btn_row, text="Adicionar", command=_add,
                  bg="#534AB7", fg="white", relief=tk.FLAT,
                  padx=10, pady=4, font=("Arial", 10), cursor="hand2"
                  ).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Remover selecionado", command=_remove,
                  bg="#7A2020", fg="white", relief=tk.FLAT,
                  padx=10, pady=4, font=("Arial", 10), cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(btn_row, text="Importar TXT", command=_import_txt,
                  bg="#1E5A4A", fg="white", relief=tk.FLAT,
                  padx=10, pady=4, font=("Arial", 10), cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(8, 0))

    def _append_text_safe(self, text, tag="translation"):
        def task():
            self.text_box.config(state=tk.NORMAL)
            self.text_box.insert(tk.END, text, tag)
            self.text_box.see(tk.END)
            self.text_box.config(state=tk.DISABLED)
        self.root.after(0, task)

    def _clear_text_safe(self):
        def task():
            self.text_box.config(state=tk.NORMAL)
            self.text_box.delete(1.0, tk.END)
            self.text_box.config(state=tk.DISABLED)
        self.root.after(0, task)

    def _set_status_safe(self, text, color="#888780"):
        self.root.after(0, lambda: self.status_lbl.config(text=text, fg=color))

    def _set_detected_safe(self, text):
        self.root.after(0, lambda: self.detected_lbl.config(text=text))

    def _set_engine_safe(self, text):
        self.root.after(0, lambda: self.engine_lbl.config(text=f"Motor: {text}"))

    def _draw_bubbles_safe(self, bubbles, active_id=None):
        def task():
            if self.show_overlay_var.get():
                self.bubble_overlay.draw_bubbles(bubbles, active_id)
        self.root.after(0, task)

    def _draw_replacements_safe(self, bubbles):
        def task():
            if self.show_overlay_var.get():
                self.bubble_overlay.draw_replacements(bubbles)
        self.root.after(0, task)

    def _set_active_safe(self, bid):
        def task():
            if self.show_overlay_var.get():
                self.bubble_overlay.set_active(bid)
        self.root.after(0, task)

    def _clear_active_safe(self):
        def task():
            if self.show_overlay_var.get():
                self.bubble_overlay.clear_active()
        self.root.after(0, task)

    def select_region(self):
        self.bubble_overlay.hide()
        self.root.withdraw()
        self.root.update()
        time.sleep(0.2)
        selector = RegionSelector(self.root)
        region   = selector.select()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

        if region:
            self.region = region
            self.dialog_history.clear()   # nova cena/capítulo: zera o contexto
            self.bubble_overlay.set_region(region)
            if not self.show_overlay_var.get():
                self.bubble_overlay.hide()
            self.region_lbl.config(
                text=f"Região: {region[0]},{region[1]} → {region[2]},{region[3]} "
                     f"({region[2]-region[0]}x{region[3]-region[1]}px)",
                fg="#9FE1CB"
            )
            self.btn_toggle.config(state=tk.NORMAL)
            self.btn_once.config(state=tk.NORMAL)
            self._append_text_safe("Região selecionada. Pronto para traduzir.\n", "info")
        else:
            self._append_text_safe("Nenhuma região selecionada.\n", "error")

    def _google_translate(self, text, source_code, target_code):
        try:
            text = text[:MAX_TRANSLATE_CHARS]
            sl   = "auto" if source_code == "auto" else source_code
            url  = (
                "https://translate.googleapis.com/translate_a/single"
                f"?client=gtx&sl={sl}&tl={target_code}&dt=t"
                f"&q={urllib.parse.quote(text)}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            if not isinstance(result, list) or not result or not result[0]:
                return "[Erro Google Translate: resposta inesperada]", "ERRO"
            translated = "".join(s[0] for s in result[0] if s and s[0])
            detected   = "—"
            if len(result) > 2 and isinstance(result[2], str):
                code     = result[2].lower()
                detected = LANG_DISPLAY.get(code, result[2].upper())
            return translated, detected
        except Exception as e:
            return f"[Erro Google Translate: {e}]", "ERRO"

    def capture_and_translate(self):
        """Ponto de entrada da tradução. Auto-protegido contra execuções
        concorrentes — uma chamada enquanto outra roda é simplesmente ignorada."""
        if not self.region:
            return
        if not self._busy_lock.acquire(blocking=False):
            return
        try:
            self._run_translation()
        finally:
            self._busy_lock.release()

    def _run_translation(self):
        try:
            self._set_status_safe("Capturando...", "#FAC775")
            # Esconde o overlay antes da captura para não fotografar a tradução
            # anterior — senão o OCR a leria como se fosse texto da página e ela
            # entraria no histórico de diálogo.
            self.root.after(0, self.bubble_overlay.hide)
            time.sleep(0.15)
            # all_screens=True captura o desktop virtual inteiro (todos os
            # monitores), não só o primário.
            screenshot = ImageGrab.grab(bbox=self.region, all_screens=True)

            src_name = self.source_lang_var.get()
            src_code = SOURCE_LANGUAGES.get(src_name, "ja")
            tgt_name = self.target_lang_var.get()
            tgt_code = TARGET_LANGUAGES.get(tgt_name, "pt")

            bubbles = self._get_bubbles(screenshot, src_code)

            timestamp = time.strftime("%H:%M:%S")
            self._clear_text_safe()

            if not bubbles:
                self._draw_bubbles_safe([])
                self._append_text_safe(f"[{timestamp}]\n", "header")
                self._append_text_safe("Nenhum texto detectado na imagem.\n", "info")
                self._set_status_safe(f"Sem texto — {timestamp}", "#888780")
                return

            self._draw_bubbles_safe(bubbles)

            translations, detected_lang = self._translate_bubbles(
                bubbles, src_code, tgt_code, tgt_name
            )
            self._set_detected_safe(detected_lang)

            replace = self.replace_var.get()
            if replace:
                # Substituição in-place: cobre o texto original e desenha a
                # tradução dentro de cada balão.
                sw, sh = screenshot.size
                render = []
                for idx, b in enumerate(bubbles):
                    bx1 = int(b["x"] / 1000 * sw)
                    by1 = int(b["y"] / 1000 * sh)
                    bx2 = int((b["x"] + b["w"]) / 1000 * sw)
                    by2 = int((b["y"] + b["h"]) / 1000 * sh)
                    bg, fg = sample_bubble_colors(screenshot, (bx1, by1, bx2, by2))
                    render.append({
                        "id": b["id"], "x": b["x"], "y": b["y"],
                        "w": b["w"], "h": b["h"],
                        "text": translations[idx], "bg": bg, "fg": fg,
                    })
                self._draw_replacements_safe(render)

            self._append_text_safe(
                f"[{timestamp}]  {src_name}  →  {tgt_name}  |  "
                f"Detectado: {detected_lang}  |  {len(bubbles)} elemento(s)\n\n",
                "header"
            )

            for idx, bubble in enumerate(bubbles):
                bid       = bubble["id"]
                btype     = bubble.get("type", "speech_bubble")
                color_tag = f"bubble_{(bid-1) % len(BUBBLE_COLORS) + 1}"
                icon      = BUBBLE_TYPE_ICONS.get(btype, "💬")

                if not replace:
                    self._set_active_safe(bid)
                self._append_text_safe(f"[{bid}] ", color_tag)
                self._append_text_safe(f"{icon} ", "type_label")
                self._append_text_safe(f"{translations[idx]}\n", "translation")
                time.sleep(0.05)

            self._append_text_safe("\n", "translation")
            if not replace:
                self._clear_active_safe()
            self._set_status_safe(f"Atualizado às {timestamp}", "#9FE1CB")

        except Exception as e:
            self._append_text_safe(f"\nErro: {str(e)}\n", "error")
            self._set_status_safe("Erro", "#F09595")

    def _get_bubbles(self, screenshot, src_code):
        """Reconhecimento unificado: detecção (comic-text-detector) para todos
        os idiomas + OCR por recorte — manga-ocr (japonês) ou EasyOCR (demais).
        """
        proc = preprocess_capture(screenshot)
        self._set_status_safe("Detectando balões...", "#9FE1CB")
        boxes = self.detector.detect(
            proc, status_cb=lambda m: self._set_status_safe(m, "#FAC775")
        )
        if not boxes:
            return []

        is_ja  = (src_code == "ja")
        engine = "manga-ocr" if is_ja else "EasyOCR"
        self._set_engine_safe(engine)

        cb  = lambda m: self._set_status_safe(m, "#FAC775")
        raw = []
        pw, ph = proc.size
        for i, (x, y, w, h) in enumerate(boxes):
            self._set_status_safe(
                f"Lendo texto {i+1}/{len(boxes)} ({engine})...", "#9FE1CB"
            )
            crop = proc.crop((x, y, x + w, y + h))
            if is_ja:
                text = self.ocr_engine.read(crop, status_cb=cb)
            else:
                text = self.easyocr_engine.read(crop, src_code, status_cb=cb)
            if not text or not is_meaningful_text(text):
                continue
            raw.append({
                "id":   i + 1,
                "type": "speech_bubble",
                "x": x / pw * 1000,
                "y": y / ph * 1000,
                "w": w / pw * 1000,
                "h": h / ph * 1000,
                "text": text,
            })
        return validate_and_fix_bubbles(raw)

    def _translate_bubbles(self, bubbles, src_code, tgt_code, tgt_name):
        """Traduz todos os balões.

        Prioridade dos motores:
          1. Marian local (Helsinki-NLP/opus-mt) — quando o par src→tgt tem
             modelo dedicado; rápido, offline após download, sem contexto.
          2. LLM local (Qwen) com contexto de diálogo — quando checkbox marcado
             e Marian falhou ou não suporta o par.
          3. Google Translate — fallback online, balão a balão.

        Retorna (lista_de_traduções, idioma_detectado).
        """
        texts = [b["text"] for b in bubbles]

        # Marian: modelo dedicado ao par — prioridade máxima quando disponível e habilitado.
        if self.use_marian_var.get() and self.marian_translator.supports(src_code, tgt_code):
            try:
                translations = self.marian_translator.translate(
                    texts, src_code, tgt_code,
                    status_cb=lambda m: self._set_status_safe(m, "#FAC775"),
                )
                self._set_engine_safe(f"Marian ({src_code}→{tgt_code})")
                return translations, LANG_DISPLAY.get(src_code, src_code)
            except Exception as e:
                self._append_text_safe(
                    f"\n[Tradução Marian indisponível: {e} — tentando LLM/Google]\n",
                    "info"
                )

        # LLM: contexto de diálogo, mais lento — quando checkbox marcado.
        if self.use_llm_var.get():
            try:
                translations = self.llm_translator.translate(
                    texts, tgt_name,
                    src_code=src_code,
                    history=list(self.dialog_history),
                    glossary=self.glossary or None,
                    status_cb=lambda m: self._set_status_safe(m, "#FAC775"),
                )
                # Alimenta o contexto da história com os novos pares.
                # Higiene: pula lixo de OCR e trunca entradas longas para o
                # histórico não estourar o prompt das páginas seguintes.
                for src, tr in zip(texts, translations):
                    if not is_meaningful_text(src):
                        continue
                    self.dialog_history.append((src[:150], tr[:150]))
                self._set_engine_safe("LLM (contexto)")
                return translations, LANG_DISPLAY.get(src_code, src_code)
            except Exception as e:
                self._append_text_safe(
                    f"\n[Tradução LLM indisponível: {e} — usando Google]\n",
                    "info"
                )

        self._set_engine_safe("Google")
        self._set_status_safe("Traduzindo (Google)...", "#FAC775")
        translations, detected = [], "—"
        for i, b in enumerate(bubbles):
            tr, det = self._google_translate(b["text"], src_code, tgt_code)
            translations.append(tr)
            if i == 0:
                detected = det
        return translations, detected

    def translate_once(self):
        if not self.region:
            return
        threading.Thread(target=self.capture_and_translate, daemon=True).start()

    def _translation_loop(self):
        while self.is_running:
            self.capture_and_translate()
            interval = self.interval_var.get()
            for _ in range(interval * 10):
                if not self.is_running:
                    break
                time.sleep(0.1)

    def toggle_translation(self):
        self.stop_translation() if self.is_running else self.start_translation()

    def start_translation(self):
        if not self.region:
            return
        self.is_running = True
        self.btn_toggle.config(text="Parar", bg="#A32D2D")
        self._set_status_safe("Rodando...", "#9FE1CB")
        threading.Thread(target=self._translation_loop, daemon=True).start()

    def stop_translation(self):
        self.is_running = False
        self.root.after(
            0, lambda: self.btn_toggle.config(text="Iniciar", bg="#3B6D11")
        )
        self._set_status_safe("Parado", "#888780")

    def run(self):
        self._append_text_safe("Universal Translator iniciado.\n", "header")
        self._append_text_safe("Pipeline 100% local — sem Ollama.\n", "info")
        self._append_text_safe("1. Selecione o idioma a ler e o idioma de saída.\n", "info")
        self._append_text_safe("2. Selecione a região da tela com o mangá.\n", "info")
        self._append_text_safe("3. OCR: manga-ocr (japonês) ou EasyOCR (demais idiomas).\n", "info")
        self._append_text_safe("4. Tradução com contexto (LLM) marcada = melhor qualidade, porém mais lenta.\n", "info")
        self._append_text_safe("5. \"Substituir texto nos balões\" = mostra a tradução no lugar do texto original.\n", "info")
        self._append_text_safe("6. F9 = traduzir 1x  |  F10 = parar modo contínuo.\n\n", "info")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.is_running = False
        keyboard.unhook_all()
        try:
            self.bubble_overlay.win.destroy()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    print("Manga Translator — pipeline 100% local (sem Ollama).")
    print("Detecção: comic-text-detector | OCR: manga-ocr / EasyOCR | "
          "Tradução: Google Translate")
    print("Iniciando interface...")

    app = OverlayWindow()
    app.run()
