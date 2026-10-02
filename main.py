from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import quote

import fitz  # PyMuPDF
import pytesseract
import requests
from PIL import Image
from rapidfuzz import fuzz
from supabase import Client, create_client


# ============================================================
# CONFIGURAZIONE E VARIABILI D'AMBIENTE
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "sources").strip()
ICCD_SPARQL_ENDPOINT = os.getenv(
    "ICCD_SPARQL_ENDPOINT",
    "https://dati.cultura.gov.it/sparql",
).strip()

TABLE_DOCUMENTS = "d27_documents"
TABLE_PAGES = "d27_document_pages"
TABLE_EVIDENCE = "d27_evidence"
TABLE_SUPPORT = "d27_evidence_support"
TABLE_ICCD = "d27_iccd_mapping_proposals"
VIEW_VALIDATED = "d27_v_validated_knowledge"

MIN_NATIVE_TEXT_CHARS = 80
MIN_SENTENCE_CHARS = 35
MAX_SENTENCE_CHARS = 900
ICCD_MIN_SCORE = 45.0
ICCD_MAX_RESULTS = 8

# Termini-segnale generali per individuare frasi storico-architettoniche.
ARCHITECTURAL_HEADS = [
    "casino", "palazzo", "belvedere", "filanda", "setificio", "chiesa",
    "quartiere", "quartieri", "fabbricato", "edificio", "corpo", "ala",
    "avancorpo", "cortile", "cocolliera", "coculliera", "opificio",
    "manifattura", "bagno", "filatoio", "tintoria", "colonia",
    "acquedotto", "officina", "magazzino", "deposito", "giardino",
]

EVENT_KEYWORDS = {
    "transformation": [
        "ampli", "riatt", "rifac", "trasform", "convert", "ricostru",
        "demol", "sopraelev", "modific", "adatt", "ristruttur", "restaur",
    ],
    "construction": [
        "costru", "edific", "erett", "realizz", "fondat", "impiant",
        "fabbricat", "innalzat",
    ],
    "function_use": [
        "destinat", "adibit", "utilizz", "uso", "funzion", "produzion",
        "filatur", "tessitur", "setificio", "vinificaz", "deposit",
        "alloggi", "scuol", "museo", "universit",
    ],
    "spatial_relation": [
        "anness", "addoss", "colleg", "adiacent", "confin", "a nord",
        "a sud", "a est", "a ovest", "presso", "accanto", "parte di",
        "interno", "esterna", "superiore", "inferiore",
    ],
    "protection_cataloguing": [
        "vincolo", "tutela", "catalog", "scheda", "particell", "catastal",
    ],
}

STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "del",
    "dello", "della", "dei", "degli", "delle", "da", "dal", "dallo",
    "dalla", "dai", "dagli", "dalle", "a", "al", "allo", "alla", "ai",
    "agli", "alle", "in", "nel", "nello", "nella", "nei", "negli", "nelle",
    "con", "su", "per", "tra", "fra", "e", "ed", "o", "ad", "reale",
}

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀÈÉÌÒÓÙ0-9])")

# Cattura denominazioni con una testa architettonica e fino a 6 parole successive.
HEADS_RE = re.compile(
    r"\b(?P<head>(?i:" + "|".join(re.escape(x) for x in ARCHITECTURAL_HEADS) + r"))\b"
    r"(?:\s+(?:(?i:di|del|dello|della|dei|degli|delle|de['’]?|san|santa)|"
    r"[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’.-]*)){0,6}"
)


@dataclass
class Fragment:
    page_id: str
    document_id: str
    document_name: str
    page_number: int
    excerpt: str
    subject: str
    evidence_type: str
    years: tuple[int, ...]


# ============================================================
# UTILITY
# ============================================================
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize(text: str) -> str:
    text = clean_space(text).lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zàèéìòóù0-9' ]+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)


def local_name(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.rstrip("/").split("/")[-1].split("#")[-1]


def supabase_client() -> Client:
    url = SUPABASE_URL
    key = SUPABASE_SERVICE_ROLE_KEY

    # Debug e controlli di validità
    print(f"[DEBUG] SUPABASE_URL estratto -> '{url}' (lunghezza: {len(url)})")
    
    if not url:
        raise ValueError("ERRORE: La variabile SUPABASE_URL è vuota o non trovata.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"ERRORE: L'URL di Supabase non è valido -> '{url}'. Deve iniziare con 'https://'")
    if not key:
        raise ValueError("ERRORE: La variabile SUPABASE_SERVICE_ROLE_KEY è vuota o non trovata.")

    return create_client(url, key)


def iter_storage_files(sb: Client, prefix: str = "") -> Iterable[str]:
    """Lista ricorsivamente i file del bucket."""
    items = sb.storage.from_(SOURCE_BUCKET).list(prefix)
    for item in items:
        name = item.get("name")
        if not name:
            continue
        path = f"{prefix}/{name}".strip("/")
        if item.get("id"):
            yield path
        else:
            yield from iter_storage_files(sb, path)


def fetch_one(sb: Client, table: str, field: str, value: Any) -> dict[str, Any] | None:
    rows = sb.table(table).select("*").eq(field, value).limit(1).execute().data or []
    return rows[0] if rows else None


# ============================================================
# 1. STORAGE -> DOCUMENTI -> PAGINE
# ============================================================
def register_and_extract_documents(sb: Client) -> None:
    paths = [
        p for p in iter_storage_files(sb)
        if p.lower().endswith((".pdf", ".txt", ".md"))
    ]

    print(f"[1/4] Fonti trovate nel bucket '{SOURCE_BUCKET}': {len(paths)}")

    for storage_path in paths:
        file_name = storage_path.rsplit("/", 1)[-1]
        print(f"  - {file_name}")

        try:
            raw = sb.storage.from_(SOURCE_BUCKET).download(storage_path)
            digest = sha256_bytes(raw)
            existing = fetch_one(sb, TABLE_DOCUMENTS, "storage_path", storage_path)

            if existing and existing.get("sha256") == digest and existing.get("processing_status") == "processed":
                print("    già processato, nessuna modifica")
                continue

            if existing:
                doc_id = existing["id"]
                sb.table(TABLE_DOCUMENTS).update({
                    "file_name": file_name,
                    "sha256": digest,
                    "processing_status": "processing",
                    "processing_error": None,
                }).eq("id", doc_id).execute()
                sb.table(TABLE_PAGES).delete().eq("document_id", doc_id).execute()
            else:
                inserted = sb.table(TABLE_DOCUMENTS).insert({
                    "storage_path": storage_path,
                    "file_name": file_name,
                    "title": os.path.splitext(file_name)[0],
                    "source_type": "pdf" if file_name.lower().endswith(".pdf") else "text",
                    "sha256": digest,
                    "processing_status": "processing",
                }).execute().data
                doc_id = inserted[0]["id"]

            pages = extract_file_pages(file_name, raw)
            for page in pages:
                page_payload = {
                    "document_id": doc_id,
                    "page_number": page["page_number"],
                    "extraction_method": page["extraction_method"],
                    "page_text": page["page_text"],
                    "char_count": len(page["page_text"] or ""),
                }
                existing_page = (
                    sb.table(TABLE_PAGES)
                    .select("id")
                    .eq("document_id", doc_id)
                    .eq("page_number", page["page_number"])
                    .limit(1)
                    .execute()
                    .data or []
                )
                if existing_page:
                    sb.table(TABLE_PAGES).update(page_payload).eq("id", existing_page[0]["id"]).execute()
                else:
                    sb.table(TABLE_PAGES).insert(page_payload).execute()

            sb.table(TABLE_DOCUMENTS).update({
                "processing_status": "processed",
                "processing_error": None,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", doc_id).execute()

        except Exception as exc:
            print(f"    ERRORE: {exc}", file=sys.stderr)
            existing = fetch_one(sb, TABLE_DOCUMENTS, "storage_path", storage_path)
            if existing:
                sb.table(TABLE_DOCUMENTS).update({
                    "processing_status": "error",
                    "processing_error": str(exc)[:2000],
                }).eq("id", existing["id"]).execute()


def extract_file_pages(file_name: str, raw: bytes) -> list[dict[str, Any]]:
    if file_name.lower().endswith((".txt", ".md")):
        text = raw.decode("utf-8", errors="replace")
        return [{
            "page_number": 1,
            "extraction_method": "plain_text",
            "page_text": clean_space(text),
        }]

    if not file_name.lower().endswith(".pdf"):
        return []

    pdf = fitz.open(stream=raw, filetype="pdf")
    out: list[dict[str, Any]] = []

    for index, page in enumerate(pdf):
        page_number = index + 1
        native = clean_space(page.get_text("text"))

        if len(native) >= MIN_NATIVE_TEXT_CHARS:
            out.append({
                "page_number": page_number,
                "extraction_method": "native_text",
                "page_text": native,
            })
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr = clean_space(pytesseract.image_to_string(image, lang="ita"))
        out.append({
            "page_number": page_number,
            "extraction_method": "ocr" if ocr else "empty",
            "page_text": ocr,
        })

    return out


# ============================================================
# 2. CROSS-READING DETERMINISTICO -> EVIDENCE AUTOMATICHE
# ============================================================
def split_sentences(text: str) -> list[str]:
    text = clean_space(text)
    if not text:
        return []
    return [clean_space(x) for x in SENTENCE_RE.split(text) if clean_space(x)]


def classify_evidence_type(sentence: str) -> str | None:
    low = sentence.lower()
    for category in [
        "transformation",
        "construction",
        "function_use",
        "spatial_relation",
        "protection_cataloguing",
    ]:
        if any(k in low for k in EVENT_KEYWORDS[category]):
            return category
    if YEAR_RE.search(sentence):
        return "chronology"
    return None


def extract_subject(sentence: str) -> str | None:
    matches = list(HEADS_RE.finditer(sentence))
    if not matches:
        return None

    candidates: list[str] = []
    for m in matches:
        candidate = clean_space(m.group(0)).strip(" ,.;:()[]")
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda x: (len(normalize(x).split()), len(x)), reverse=True)
    return candidates[0]


def extract_fragments(sb: Client) -> list[Fragment]:
    pages = (
        sb.table(TABLE_PAGES)
        .select(f"id,document_id,page_number,page_text,{TABLE_DOCUMENTS}(file_name)")
        .execute()
        .data or []
    )

    fragments: list[Fragment] = []
    for page in pages:
        doc_info = page.get(TABLE_DOCUMENTS) or {}
        doc_name = doc_info.get("file_name") or page["document_id"]
        for sentence in split_sentences(page.get("page_text") or ""):
            if not (MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS):
                continue

            evidence_type = classify_evidence_type(sentence)
            if not evidence_type:
                continue

            subject = extract_subject(sentence)
            if not subject:
                continue

            years = tuple(sorted({int(y) for y in YEAR_RE.findall(sentence)}))
            fragments.append(Fragment(
                page_id=page["id"],
                document_id=page["document_id"],
                document_name=doc_name,
                page_number=int(page["page_number"]),
                excerpt=sentence,
                subject=subject,
                evidence_type=evidence_type,
                years=years,
            ))

    return fragments


def canonical_subject(subject: str, canonicals: list[str]) -> str:
    n = normalize(subject)
    if not canonicals:
        canonicals.append(subject)
        return subject

    best = None
    best_score = 0.0
    for existing in canonicals:
        score = float(fuzz.token_set_ratio(n, normalize(existing)))
        if score > best_score:
            best = existing
            best_score = score

    if best is not None and best_score >= 88.0:
        if len(normalize(subject).split()) > len(normalize(best).split()):
            idx = canonicals.index(best)
            canonicals[idx] = subject
            return subject
        return best

    canonicals.append(subject)
    return subject


def build_cross_reading_evidence(sb: Client) -> None:
    print("[2/4] Cross-reading automatico delle fonti")
    fragments = extract_fragments(sb)
    print(f"    frammenti candidati: {len(fragments)}")

    canonicals: list[str] = []
    grouped: dict[tuple[str, str, str], list[Fragment]] = defaultdict(list)

    for fragment in fragments:
        subject = canonical_subject(fragment.subject, canonicals)
        year_key = ",".join(str(y) for y in fragment.years) if fragment.years else "undated"
        key = (normalize(subject), fragment.evidence_type, year_key)
        grouped[key].append(fragment)

    touched_evidence_ids: set[str] = set()

    for (_, evidence_type, _), group in grouped.items():
        subject = max((f.subject for f in group), key=lambda x: (len(normalize(x).split()), len(x)))
        years = sorted({y for f in group for y in f.years})
        start_year = years[0] if years else None
        end_year = years[-1] if years else None
        documents = {f.document_id for f in group}
        support_count = len(group)
        document_count = len(documents)
        cross_source = document_count >= 2

        representative = sorted(
            group,
            key=lambda f: (0 if f.years else 1, len(f.excerpt))
        )[0].excerpt

        confidence = 0.50
        if document_count >= 2:
            confidence += 0.15
        if document_count >= 3:
            confidence += 0.10
        if support_count >= 3:
            confidence += 0.05
        if years:
            confidence += 0.05
        confidence = min(confidence, 0.95)

        fingerprint_basis = "|".join([
            normalize(subject),
            evidence_type,
            str(start_year or ""),
            str(end_year or ""),
        ])
        fingerprint = sha256_text(fingerprint_basis)

        existing = fetch_one(sb, TABLE_EVIDENCE, "fingerprint", fingerprint)
        auto_payload = {
            "fingerprint": fingerprint,
            "subject_auto": subject,
            "evidence_type_auto": evidence_type,
            "start_year_auto": start_year,
            "end_year_auto": end_year,
            "representative_excerpt": representative,
            "support_count": support_count,
            "document_count": document_count,
            "cross_source": cross_source,
            "confidence_auto": round(confidence, 3),
            "generation_method": "deterministic_cross_reading_v1",
        }

        if existing:
            evidence_id = existing["id"]
            sb.table(TABLE_EVIDENCE).update(auto_payload).eq("id", evidence_id).execute()
            sb.table(TABLE_SUPPORT).delete().eq("evidence_id", evidence_id).execute()
        else:
            inserted = sb.table(TABLE_EVIDENCE).insert(auto_payload).execute().data
            evidence_id = inserted[0]["id"]

        touched_evidence_ids.add(evidence_id)

        supports = []
        seen_supports: set[tuple[str, str]] = set()
        for f in group:
            excerpt_hash = sha256_text(clean_space(f.excerpt))
            dedupe_key = (f.page_id, excerpt_hash)
            if dedupe_key in seen_supports:
                continue
            seen_supports.add(dedupe_key)
            supports.append({
                "evidence_id": evidence_id,
                "page_id": f.page_id,
                "source_excerpt": f.excerpt,
                "excerpt_hash": excerpt_hash,
            })

        if supports:
            sb.table(TABLE_SUPPORT).insert(supports).execute()

    print(f"    evidence automatiche generate/aggiornate: {len(grouped)}")


# ============================================================
# 3. EVIDENCE VALIDATE -> ICCD / ArCo (READ ONLY)
# ============================================================
def sparql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def make_query_terms(subject: str) -> list[str]:
    subject = clean_space(subject)
    terms = [subject]

    tokens = [t for t in normalize(subject).split() if len(t) >= 4]
    for token in sorted(tokens, key=len, reverse=True)[:2]:
        if token.lower() not in {x.lower() for x in terms}:
            terms.append(token)
    return terms


def query_iccd(term: str) -> list[dict[str, Any]]:
    safe = sparql_escape(term)
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX a-cat: <https://w3id.org/arco/ontology/catalogue/>

SELECT DISTINCT ?item ?label ?record ?recordType ?nct
WHERE {{
  ?item rdfs:label ?label .
  ?record a-cat:describesCulturalProperty ?item ;
          a-cat:catalogueRecordIdentifier ?nct ;
          a ?recordType .

  FILTER(lang(?label) = "" || langMatches(lang(?label), "it"))
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{safe}")))
  FILTER(CONTAINS(STR(?recordType), "CatalogueRecord"))
}}
LIMIT 30
"""

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "DOTATI2027-SanLeucio/1.0 academic prototype",
    }
    response = requests.get(
        ICCD_SPARQL_ENDPOINT,
        params={"query": query, "format": "application/sparql-results+json"},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    bindings = response.json().get("results", {}).get("bindings", [])

    out: list[dict[str, Any]] = []
    for b in bindings:
        item_uri = b.get("item", {}).get("value")
        label = b.get("label", {}).get("value")
        record_uri = b.get("record", {}).get("value")
        record_type_uri = b.get("recordType", {}).get("value")
        nct = b.get("nct", {}).get("value")
        if not item_uri or not label:
            continue

        item_kind = None
        if "/resource/" in item_uri:
            tail = item_uri.split("/resource/", 1)[1]
            item_kind = tail.split("/", 1)[0] if "/" in tail else None

        catalogue_url = None
        if item_kind and nct:
            catalogue_url = f"https://catalogo.beniculturali.it/detail/{quote(item_kind)}/{quote(nct)}"

        out.append({
            "item_uri": item_uri,
            "label": label,
            "record_uri": record_uri,
            "record_type": local_name(record_type_uri),
            "nct": nct,
            "catalogue_url": catalogue_url,
            "raw": b,
        })
    return out


def label_score(query_term: str, label: str) -> tuple[str, float]:
    q = normalize(query_term)
    l = normalize(label)
    score = float(fuzz.token_set_ratio(q, l))
    if q == l:
        return "exact", 100.0
    if score >= 85:
        return "strong", score
    return "partial", score


def lookup_iccd_for_validated_evidence(sb: Client) -> None:
    rows = sb.table(VIEW_VALIDATED).select("*").execute().data or []
    print(f"[3/4] Evidence validate disponibili per il confronto ICCD: {len(rows)}")

    for row in rows:
        evidence_id = row["evidence_id"]
        subject = clean_space(row.get("subject") or "")
        if not subject:
            continue

        existing = (
            sb.table(TABLE_ICCD)
            .select("id")
            .eq("evidence_id", evidence_id)
            .limit(1)
            .execute()
            .data or []
        )
        if existing:
            continue

        candidates_by_uri: dict[str, dict[str, Any]] = {}
        query_terms = make_query_terms(subject)

        for term in query_terms:
            try:
                matches = query_iccd(term)
            except Exception as exc:
                print(f"    ICCD lookup fallito per '{term}': {exc}", file=sys.stderr)
                continue

            for match in matches:
                kind, score = label_score(subject, match["label"])
                if score < ICCD_MIN_SCORE:
                    continue
                key = match["item_uri"]
                old = candidates_by_uri.get(key)
                candidate = {**match, "match_type": kind, "score": score, "query_term": term}
                if old is None or candidate["score"] > old["score"]:
                    candidates_by_uri[key] = candidate

            time.sleep(0.15)

        ranked = sorted(candidates_by_uri.values(), key=lambda x: x["score"], reverse=True)[:ICCD_MAX_RESULTS]

        if not ranked:
            mapping_fingerprint = sha256_text(f"{evidence_id}|NO_MATCH|{subject}")
            sb.table(TABLE_ICCD).insert({
                "mapping_fingerprint": mapping_fingerprint,
                "evidence_id": evidence_id,
                "query_term": subject,
                "match_found": False,
                "match_type": "none",
                "similarity_score": 0,
                "raw_payload": {"searched_terms": query_terms},
            }).execute()
            print(f"    {subject}: nessuna scheda pubblica candidata individuata")
            continue

        for candidate in ranked:
            mapping_fingerprint = sha256_text(
                f"{evidence_id}|{candidate['item_uri']}|{candidate.get('nct') or ''}"
            )
            sb.table(TABLE_ICCD).insert({
                "mapping_fingerprint": mapping_fingerprint,
                "evidence_id": evidence_id,
                "query_term": candidate["query_term"],
                "match_found": True,
                "iccd_label": candidate["label"],
                "nct": candidate.get("nct"),
                "record_type": candidate.get("record_type"),
                "item_uri": candidate["item_uri"],
                "catalogue_url": candidate.get("catalogue_url"),
                "match_type": candidate["match_type"],
                "similarity_score": round(candidate["score"], 2),
                "raw_payload": candidate.get("raw"),
            }).execute()

        print(f"    {subject}: {len(ranked)} candidati ICCD salvati")


# ============================================================
# 4. RIEPILOGO E MAIN
# ============================================================
def print_summary(sb: Client) -> None:
    print("[4/4] Riepilogo")
    for table, label in [
        (TABLE_DOCUMENTS, "documenti"),
        (TABLE_PAGES, "pagine"),
        (TABLE_EVIDENCE, "evidence"),
        (TABLE_ICCD, "mapping ICCD"),
    ]:
        result = sb.table(table).select("id", count="exact").limit(1).execute()
        count = result.count if result.count is not None else "?"
        print(f"    {label}: {count}")


def main() -> None:
    sb = supabase_client()
    register_and_extract_documents(sb)
    build_cross_reading_evidence(sb)
    lookup_iccd_for_validated_evidence(sb)
    print_summary(sb)


if __name__ == "__main__":
    main()
