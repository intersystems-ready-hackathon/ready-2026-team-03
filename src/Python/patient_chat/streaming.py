import json
import logging
import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator
import streamlit as st
from openai import APIError, RateLimitError
from agent import PatientAgent

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def random_caption(agent_name: str = "CareBot") -> str:
    """Return a random caption from the captions list."""
    # import captions from assets/captions.json using Path
    parent_dir = Path(__file__).parent
    with (parent_dir / "assets" / "captions.json").open("r") as f:
        CAPTIONS = json.load(f)
    return f"✨ {agent_name} {random.choice(CAPTIONS)}"

def stream_wrapper(query: str, info_container: dict, agent: PatientAgent) -> Iterator[str]:
    """Stream the agent's response token-by-token for `st.write_stream`. """
    info_container["active_agent"] = "carebot"
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    history.append({"role": "user", "content": query})
    logger.info("New query received (length=%d, history_turns=%d)", len(query), len(history) - 1)
    try:
        gen = agent.stream(history)
        # Real Streamlit spinner while we wait for the first token (most of the
        # latency lives here). As soon as content starts arriving we drop the
        # spinner so the response can build up in real time.
        with st.spinner(random_caption("CareBot"), show_time=True):
            first_chunk = next(gen, None)
        if first_chunk is not None:
            yield first_chunk
            for chunk in gen:
                yield chunk
    except RateLimitError as exc:
        info_container["error"] = str(exc)
        logger.warning("OpenAI rate limit / quota error: %s", exc)
        yield (
            "\n\n**The model rejected the request (rate limit / quota).** "
            f"Details: `{exc.code or 'rate_limit'}`. "
            "Check your OpenAI project's model access and billing, "
            "or set `OPENAI_MODEL` in `.env` to a model your key can call "
            "(e.g. `gpt-5-nano`, `gpt-4o-mini`)."
        )
    except APIError as exc:
        info_container["error"] = str(exc)
        logger.exception("OpenAI API error during streaming")
        yield f"\n\n**OpenAI API error:** {exc.message or exc}"
    finally:
        info_container["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.debug("stream_wrapper finished at %s", info_container["timestamp"])
