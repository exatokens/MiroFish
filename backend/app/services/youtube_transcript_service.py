"""YouTube transcript extraction service for MiroFish.

Accepts a plain-text file where each line is a YouTube URL.
Fetches captions via youtube_transcript_api (no API key required),
runs LLM extraction per video to pull structured investment signals,
then returns a formatted document string ready to be fed into the
knowledge graph as a reference document.
"""

import json
import re
import textwrap
from typing import Optional

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.youtube_transcript")

# Words per transcript chunk sent to LLM (keep low to stay within 8K context window)
CHUNK_WORDS = 1500
# Max chunks per video (caps token spend)
MAX_CHUNKS = 6

# YouTube URL pattern — matches watch?v=, youtu.be/, shorts/, embed/
_YT_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


# ── URL detection & parsing ────────────────────────────────────────────────────

def extract_ticker_from_text(text: str) -> str:
    """Extract ticker symbol from a comment line in the file content.

    Supports formats like:
        # TICKER: ASTS
        # ticker: asts
        # ASTS

    Args:
        text: Raw file content.

    Returns:
        Uppercase ticker string, or empty string if not found.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break  # Only inspect leading comment lines
        # Match "# TICKER: ASTS" or "# ticker: asts" or "# ASTS"
        m = re.match(r"#\s*(?:ticker[:\s]+)?([A-Z]{1,6})\s*$", stripped, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def is_youtube_url_list(text: str) -> bool:
    """Return True if text looks like a list of YouTube URLs.

    Args:
        text: Raw file content.

    Returns:
        True when at least one line contains a YouTube URL and ≥50 % of
        non-blank lines are YouTube URLs.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    yt_lines = sum(1 for l in lines if _YT_URL_RE.search(l))
    return yt_lines > 0 and (yt_lines / len(lines)) >= 0.5


def extract_youtube_urls(text: str) -> list[str]:
    """Extract all YouTube URLs from text (one per line or inline).

    Args:
        text: Raw file content.

    Returns:
        Deduplicated list of full YouTube URL strings in order of appearance.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept either bare URL or URL embedded in a longer line
        m = _YT_URL_RE.search(line)
        if m:
            # Reconstruct a clean watch URL from the video ID
            vid_id = m.group(1)
            url = f"https://www.youtube.com/watch?v={vid_id}"
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character video ID from any YouTube URL form.

    Args:
        url: YouTube URL string.

    Returns:
        11-char video ID, or None if the URL is not recognised.
    """
    m = _YT_URL_RE.search(url)
    return m.group(1) if m else None


# ── Transcript fetching ────────────────────────────────────────────────────────

def get_transcript(video_id: str) -> str:
    """Fetch and join transcript segments for a YouTube video.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        Full transcript as a single whitespace-joined string.

    Raises:
        Exception: If no captions are available for the video.
    """
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

    transcript = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(s.text for s in transcript)


def _chunk_text(text: str) -> list[str]:
    """Split text into word-count-bounded chunks.

    Args:
        text: Full transcript text.

    Returns:
        List of chunk strings (at most MAX_CHUNKS).
    """
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), CHUNK_WORDS):
        chunks.append(" ".join(words[i : i + CHUNK_WORDS]))
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks


# ── LLM extraction ─────────────────────────────────────────────────────────────

def _extraction_prompt(chunk: str, ticker: str, chunk_idx: int, total: int) -> str:
    """Build the per-chunk signal extraction prompt.

    Args:
        chunk:      Transcript chunk text.
        ticker:     Stock ticker to focus on (may be empty string for general).
        chunk_idx:  0-based chunk index.
        total:      Total number of chunks for this video.

    Returns:
        Full prompt string.
    """
    focus = f"Focus on {ticker}." if ticker else "Extract signals for any meaningfully discussed stock."
    return textwrap.dedent(f"""
        Extract investment signals from this transcript chunk ({chunk_idx + 1} of {total}).
        {focus}

        TRANSCRIPT:
        {chunk}

        Respond with a JSON object using these exact keys:
        ticker, bullishSignals, bearishSignals, thesis, catalysts, risks,
        priceTargets, cashFlowMentions, customerMentions, keyQuotes, macroThemes.

        - thesis: one-sentence core investment argument
        - priceTargets: list of objects with scenario (bull/base/bear), target ($XXX), rationale
        - keyQuotes: list of objects with quote and context
        - All other fields: arrays of strings
        - Use empty arrays for keys with no relevant content
    """).strip()


def _synthesis_prompt(ticker: str, video_summaries: list[dict]) -> str:
    """Build the cross-video synthesis prompt.

    Args:
        ticker:          Stock ticker.
        video_summaries: List of per-video extraction dicts.

    Returns:
        Full prompt string.
    """
    return textwrap.dedent(f"""
        Synthesise investment insights from these per-video extracts about {ticker}.

        PER-VIDEO DATA:
        {json.dumps(video_summaries, indent=2)}

        Respond with a JSON object using these exact keys:
        overallSentiment (bullish/bearish/mixed),
        sentimentRationale (2-3 sentences),
        consensusThesis (dominant thesis across all videos),
        keyBullPoints (array of strings),
        keyBearPoints (array of strings),
        priceTargetRange (object: low, high, note),
        keyCatalysts (array of strings),
        keyRisks (array of strings),
        cashFlowSummary (string),
        customerInsights (string),
        macroThemes (array of strings),
        analystConsensus (3-4 sentence synthesis about {ticker}).
    """).strip()


def _call_llm_json(prompt: str, max_retries: int = 3) -> dict:
    """Call the configured LLM and parse JSON response, with retry logic.

    Args:
        prompt:      Full prompt string.
        max_retries: Number of attempts before raising.

    Returns:
        Parsed dict from the model response.

    Raises:
        ValueError: If all attempts fail to produce valid JSON.
    """
    client = OpenAI(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
    )

    last_error: Exception = ValueError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial analyst assistant. "
                            "Always respond with a single valid JSON object only. "
                            "No markdown, no code fences, no explanation — just raw JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.1,
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            raw = (choice.message.content or "").strip()
            logger.debug(
                f"LLM attempt {attempt}: finish_reason={finish_reason}, "
                f"len={len(raw)}, preview={raw[:200]}"
            )

            if not raw:
                logger.warning(
                    f"LLM returned empty response on attempt {attempt} "
                    f"(finish_reason={finish_reason})"
                )
                last_error = ValueError(f"LLM returned empty response (finish_reason={finish_reason})")
                continue

            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                end = next(
                    (i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"),
                    len(lines),
                )
                raw = "\n".join(lines[1:end]).strip()

            # Attempt to extract a JSON object if the model added surrounding text
            if not raw.startswith("{"):
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    raw = m.group(0).strip()

            return json.loads(raw)

        except json.JSONDecodeError as exc:
            last_error = ValueError(f"LLM response is not valid JSON: {exc}\nRaw: {raw[:300]}")
            logger.warning(f"JSON parse failed attempt {attempt}: {exc}")
        except Exception as exc:
            last_error = exc
            logger.warning(f"LLM call failed attempt {attempt}: {exc}")

    raise last_error


# ── Main entry point ───────────────────────────────────────────────────────────

def process_youtube_url_list(text: str, ticker: str = "") -> str:
    """Process a text file containing YouTube URLs and return formatted document text.

    For each URL:
      1. Extract the video ID
      2. Fetch the transcript (skip if captions unavailable)
      3. Chunk the transcript and run LLM extraction per chunk
      4. Merge chunk results for the video

    Then run cross-video synthesis and format everything into a structured
    plain-text document ready to be fed into the MiroFish knowledge graph.

    Args:
        text:   Raw content of the uploaded .txt file (one YouTube URL per line).
        ticker: Optional stock ticker to focus extraction on (e.g. "NVDA").

    Returns:
        Formatted document string containing transcript insights and key quotes.
    """
    urls = extract_youtube_urls(text)
    if not urls:
        return ""

    logger.info(f"YouTube transcript extraction: {len(urls)} URLs, ticker={ticker or 'general'}")

    video_results: list[dict] = []

    for url in urls:
        vid_id = extract_video_id(url)
        if not vid_id:
            logger.warning(f"Could not parse video ID from: {url}")
            continue

        # Fetch transcript
        try:
            transcript = get_transcript(vid_id)
        except Exception as e:
            logger.warning(f"No transcript for {vid_id}: {e}")
            continue

        word_count = len(transcript.split())
        logger.info(f"Video {vid_id}: {word_count} words — extracting signals")

        chunks = _chunk_text(transcript)
        chunk_extracts: list[dict] = []

        for i, chunk in enumerate(chunks):
            try:
                prompt = _extraction_prompt(chunk, ticker, i, len(chunks))
                extract = _call_llm_json(prompt)
                chunk_extracts.append(extract)
            except Exception as e:
                logger.warning(f"Chunk {i+1}/{len(chunks)} extraction failed for {vid_id}: {e}")

        if not chunk_extracts:
            continue

        # Merge chunk results (deduplicate list values)
        def _merge(key: str) -> list:
            seen_vals: set = set()
            out: list = []
            for cr in chunk_extracts:
                for item in cr.get(key, []):
                    k = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
                    if k not in seen_vals:
                        seen_vals.add(k)
                        out.append(item)
            return out

        def _first_str(key: str) -> str:
            for cr in chunk_extracts:
                val = cr.get(key, "")
                if val:
                    return val
            return ""

        video_results.append({
            "videoId": vid_id,
            "url": url,
            "wordCount": word_count,
            "thesis": _first_str("thesis"),
            "bullishSignals": _merge("bullishSignals"),
            "bearishSignals": _merge("bearishSignals"),
            "catalysts": _merge("catalysts"),
            "risks": _merge("risks"),
            "priceTargets": _merge("priceTargets"),
            "cashFlowMentions": _merge("cashFlowMentions"),
            "customerMentions": _merge("customerMentions"),
            "keyQuotes": _merge("keyQuotes"),
            "macroThemes": _merge("macroThemes"),
        })

    if not video_results:
        return "YouTube transcript extraction: no transcripts could be retrieved from the provided URLs."

    # Cross-video synthesis
    synthesis: dict = {}
    if len(video_results) > 0:
        try:
            synth_prompt = _synthesis_prompt(ticker or "the company", video_results)
            synthesis = _call_llm_json(synth_prompt)
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

    # Format as a structured document for the knowledge graph
    lines: list[str] = []
    lines.append(f"# YouTube Video Transcript Analysis")
    if ticker:
        lines.append(f"## Ticker: {ticker}")
    lines.append(f"## Source: {len(video_results)} video(s) with transcripts extracted")
    lines.append("")

    if synthesis:
        lines.append("## Cross-Video Synthesis")
        lines.append(f"**Overall Sentiment:** {synthesis.get('overallSentiment', 'N/A')}")
        lines.append(f"**Rationale:** {synthesis.get('sentimentRationale', '')}")
        lines.append(f"**Consensus Thesis:** {synthesis.get('consensusThesis', '')}")
        lines.append("")

        pt = synthesis.get("priceTargetRange", {})
        if pt:
            lines.append(f"**Price Target Range:** {pt.get('low', 'N/A')} – {pt.get('high', 'N/A')}  ({pt.get('note', '')})")

        if synthesis.get("cashFlowSummary"):
            lines.append(f"**Cash Flow / Margins:** {synthesis['cashFlowSummary']}")

        if synthesis.get("customerInsights"):
            lines.append(f"**Customer Insights:** {synthesis['customerInsights']}")

        if synthesis.get("analystConsensus"):
            lines.append(f"**Analyst Consensus:** {synthesis['analystConsensus']}")

        lines.append("")
        for label, key in [("Key Bull Points", "keyBullPoints"), ("Key Bear Points", "keyBearPoints"),
                            ("Key Catalysts", "keyCatalysts"), ("Key Risks", "keyRisks"),
                            ("Macro Themes", "macroThemes")]:
            items = synthesis.get(key, [])
            if items:
                lines.append(f"**{label}:**")
                for item in items:
                    lines.append(f"  - {item}")
        lines.append("")

    lines.append("## Per-Video Details")
    for vr in video_results:
        lines.append(f"### Video: {vr['url']} ({vr['wordCount']:,} words)")
        if vr.get("thesis"):
            lines.append(f"**Thesis:** {vr['thesis']}")
        for label, key in [("Bullish Signals", "bullishSignals"), ("Bearish Signals", "bearishSignals"),
                            ("Catalysts", "catalysts"), ("Risks", "risks"),
                            ("Cash Flow / Margin Mentions", "cashFlowMentions"),
                            ("Customer Mentions", "customerMentions"),
                            ("Macro Themes", "macroThemes")]:
            items = vr.get(key, [])
            if items:
                lines.append(f"**{label}:**")
                for item in items:
                    lines.append(f"  - {item}")
        pts = vr.get("priceTargets", [])
        if pts:
            lines.append("**Price Targets Mentioned:**")
            for pt in pts:
                if isinstance(pt, dict):
                    lines.append(f"  - {pt.get('scenario', '').capitalize()}: {pt.get('target', 'N/A')} — {pt.get('rationale', '')}")
        quotes = vr.get("keyQuotes", [])
        if quotes:
            lines.append("**Key Quotes:**")
            for q in quotes:
                if isinstance(q, dict):
                    lines.append(f'  > "{q.get("quote", "")}"  — {q.get("context", "")}')
        lines.append("")

    return "\n".join(lines)