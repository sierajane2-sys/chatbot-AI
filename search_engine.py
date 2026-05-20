"""
BM25 기반 한국어 검색 엔진 (AI API 없이 고품질 답변)
"""

import json
import re
import math
from pathlib import Path
from collections import Counter


# ── 한국어 토크나이저 ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    한국어 텍스트 토크나이즈:
    1) 공백 분리 어절
    2) 한글 2~4자 문자 바이그램 (형태소 근사)
    3) 영숫자 단어 (대소문자 구별 없이 소문자 통일)
    """
    text = text.lower()  # 영문 대소문자 통일
    tokens = []

    # 어절 단위
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text)
    tokens.extend(words)

    # 한글 바이그램 (2글자)
    korean_only = re.sub(r"[^가-힣]", "", text)
    for i in range(len(korean_only) - 1):
        tokens.append(korean_only[i:i+2])

    # 한글 트라이그램 (3글자)
    for i in range(len(korean_only) - 2):
        tokens.append(korean_only[i:i+3])

    return tokens


# ── BM25 구현 ────────────────────────────────────────────────────

class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(d) for d in docs]
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.N, 1)
        self._build_index()

    def _build_index(self):
        self.df = {}
        self.idf = {}
        self.tf = []

        for doc in self.corpus:
            freq = Counter(doc)
            self.tf.append(freq)
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        tf = self.tf[doc_idx]
        dl = len(self.corpus[doc_idx])
        score = 0.0

        for term in q_tokens:
            if term not in self.idf:
                continue
            f = tf.get(term, 0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * numerator / max(denominator, 1e-9)

        return score

    def search(self, query: str, top_k=5) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in scores[:top_k] if sc > 0]


# ── 콘텐츠 정제 ──────────────────────────────────────────────────

_CIRCLE_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

# 단계 표현 패턴
_STEP_RE = re.compile(r"^(\d+단계\)|\d+단계\s*[).]\s*)")

# 섹션 제목처럼 보이는 짧은 줄
_SECTION_TITLE_RE = re.compile(r"^[가-힣A-Za-z\s\(\)\/]{4,25}$")


def clean_content(text: str) -> str:
    """인라인 메타데이터 제거 + 원문자·단계 앞 줄바꿈 삽입"""

    # 1. 푸터 시작 마커를 찾아 이후 내용을 통째로 제거
    FOOTER_MARKERS = [
        "매우 만족",
        "매우만족",
        "원하시는 정보를 못 찾으셨다면",
        "무상 보증기간 이내라도",
        "AI CS Bot 상담",
        "전화상담 예약",
        "이 정보가 도움이 되었나요",
        "서비스 만족도",
    ]
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # 2. 인라인 메타데이터 제거
    text = re.sub(r"\s*날짜\s*\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\s*조회수\s*[\d,]+", "", text)
    text = re.sub(r"\s*(공유하기|스크랩|인쇄|목록으로|본문 바로가기|메뉴 바로가기)", "", text)
    text = re.sub(r"\s*(이전\s*글|다음\s*글|이전화면)", "", text)
    text = re.sub(r"Samsung Electronics.*", "", text, flags=re.IGNORECASE)

    # 3. 혹시 남아있는 평가 문구 개별 제거
    remove_phrases = [
        r"무상\s*보증기간[^.]*부과됩니다\.?",
        r"원하시는 정보를 못 찾으셨다면[^.]*세요\.?",
        r"매우\s*불만족",
        r"매우\s*만족",
        r"불만족",
        r"평가하기",
        r"이전화면",
        r"AI\s*CS\s*Bot\s*상담",
        r"전화상담\s*예약",
    ]
    for phrase in remove_phrases:
        text = re.sub(phrase, "", text)

    # 2. 원문자(①②③…) 앞에 줄바꿈 삽입 (인라인 붙어있는 경우 분리)
    text = re.sub(f"([{_CIRCLE_NUMS}])", r"\n\1", text)

    # 3. 숫자 단계(1단계) 앞에 줄바꿈 삽입
    text = re.sub(r"(\d+단계[).]\s*)", r"\n\1", text)

    # 4. 줄별 정리 (빈 줄·너무 짧은 줄 제거)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) <= 2:
            continue
        lines.append(line)

    return "\n".join(lines)


def format_for_display(text: str) -> str:
    """고객 시인성을 위해 마크다운 단락 구조로 변환"""
    lines = text.splitlines()
    result = []

    for i, line in enumerate(lines):
        # 원문자로 시작하는 줄 → 앞에 빈 줄 추가
        if line and line[0] in _CIRCLE_NUMS:
            if result and result[-1] != "":
                result.append("")
            result.append(line)
            continue

        # 숫자 단계 표현 → 앞에 빈 줄 추가
        if _STEP_RE.match(line):
            if result and result[-1] != "":
                result.append("")
            result.append(line)
            continue

        # 섹션 제목처럼 보이는 줄 → 볼드 + 앞뒤 빈 줄
        if i > 0 and _SECTION_TITLE_RE.match(line):
            if result and result[-1] != "":
                result.append("")
            result.append(f"**{line}**")
            result.append("")
            continue

        result.append(line)

    # 연속 빈 줄 → 하나로 합치기
    final = []
    prev_blank = False
    for line in result:
        if line == "":
            if not prev_blank:
                final.append("")
            prev_blank = True
        else:
            final.append(line)
            prev_blank = False

    return "\n\n".join(ln for ln in "\n".join(final).split("\n\n") if ln.strip())


def extract_best_paragraphs(content: str, query: str, max_chars=1200) -> str:
    """콘텐츠에서 쿼리와 가장 관련 높은 단락을 추출 후 포맷"""
    cleaned = clean_content(content)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if len(p.strip()) > 20]

    if not paragraphs:
        return format_for_display(cleaned[:max_chars])

    q_tokens = set(tokenize(query))
    scored = []
    for p in paragraphs:
        p_tokens = set(tokenize(p))
        overlap = len(q_tokens & p_tokens)
        scored.append((overlap, len(p), p))

    scored.sort(key=lambda x: (-x[0], -x[1]))

    result = []
    total = 0
    for _, length, para in scored:
        if total + length > max_chars:
            break
        result.append(para)
        total += length

    if not result:
        result = [paragraphs[0]]

    combined = "\n\n".join(result)
    return format_for_display(combined)


def highlight_keywords(text: str, query: str) -> str:
    """쿼리 키워드를 마크다운 볼드로 강조"""
    words = re.findall(r"[가-힣]{2,}|[A-Za-z0-9]{2,}", query)
    for word in sorted(words, key=len, reverse=True):
        pattern = re.compile(re.escape(word))
        text = pattern.sub(f"**{word}**", text, count=3)
    return text


# ── 검색 엔진 메인 클래스 ─────────────────────────────────────────

class SamsungSVCSearchEngine:
    def __init__(self, data_path="data/solutions.json"):
        self.solutions = []
        self.bm25 = None
        self._load(data_path)

    def _load(self, path):
        p = Path(path)
        if not p.exists():
            return

        with open(p, "r", encoding="utf-8") as f:
            self.solutions = json.load(f)

        # 검색 대상 텍스트: 제목 × 3 (가중치) + 본문
        docs = [
            (s.get("title", "") + " ") * 3 + s.get("content", "")
            for s in self.solutions
        ]
        self.bm25 = BM25(docs)

    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.solutions) > 0

    def answer(self, query: str, top_k=3) -> dict:
        """
        질문에 대한 답변 생성
        반환: {answer, sources, found}
        """
        if not self.is_ready():
            return {
                "answer": "데이터가 아직 로딩되지 않았습니다. scraper.py를 먼저 실행해주세요.",
                "sources": [],
                "found": False,
            }

        results = self.bm25.search(query, top_k=top_k)

        if not results:
            return {
                "answer": "죄송합니다. 관련된 정보를 찾지 못했습니다. 다른 키워드로 질문해 보세요.",
                "sources": [],
                "found": False,
            }

        # 최상위 결과로 답변 구성
        best_idx, best_score = results[0]
        best = self.solutions[best_idx]

        paragraph = extract_best_paragraphs(
            best.get("content", ""), query, max_chars=1200
        )
        highlighted = highlight_keywords(paragraph, query)

        # 답변 포맷
        answer_lines = [f"### {best.get('title', '')}\n\n", highlighted]

        # 추가 관련 항목이 있으면 요약 추가
        if len(results) > 1:
            answer_lines.append("\n\n---\n\n**관련 내용**\n")
            for idx, sc in results[1:]:
                sol = self.solutions[idx]
                answer_lines.append(f"- [{sol.get('title', '')}]({sol.get('url', '')})")

        sources = []
        for idx, sc in results:
            sol = self.solutions[idx]
            sources.append({
                "title": sol.get("title", ""),
                "url": sol.get("url", ""),
                "score": round(sc, 2),
            })

        return {
            "answer": "\n".join(answer_lines),
            "sources": sources,
            "found": True,
        }
