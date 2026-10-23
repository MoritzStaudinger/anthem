import os
from itertools import product

import pandas as pd
import streamlit as st
import requests
import altair as alt
import json
import fitz  # PyMuPDF

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(layout="wide")


# --- Chart helpers ---------------------------------------------------------
# Each helper is a no-op when its required field is missing so cached results
# from older runs (or agent runs without institution profiles) render cleanly.

def _summary_row(profile):
    pubs_per_year = profile.get("PublicationsPerYear") or {}
    growth = "—"
    if pubs_per_year:
        years_int = sorted(int(y) for y in pubs_per_year.keys())
        if len(years_int) >= 6:
            def _at(y):
                return pubs_per_year.get(str(y), pubs_per_year.get(y, 0))
            recent = sum(_at(y) for y in years_int[-3:])
            prior = sum(_at(y) for y in years_int[-6:-3])
            if prior:
                growth = f"{(recent / prior - 1) * 100:+.0f}%"

    top_topics = profile.get("TopicDistribution") or []
    top_topic = top_topics[0][0] if top_topics else "—"

    collabs = profile.get("Collaborator_Universities") or []
    top_collab = collabs[0][0] if collabs else "—"

    researchers = profile.get("ResearchersWithPublication") or {}
    if isinstance(researchers, dict) and researchers:
        top_author = max(researchers.items(), key=lambda kv: kv[1])[0]
    else:
        top_author = "—"

    return {
        "Institution": profile.get("Institution", "—"),
        "Publications": profile.get("Publications", 0),
        "Period": f"{profile.get('Year Start', '?')}–{profile.get('Year End', '?')}",
        "Growth (last 3y vs prior 3y)": growth,
        "Top author": top_author,
        "Top topic": top_topic,
        "Top collaborator": top_collab,
    }


def render_summary_table(profiles):
    rows = [_summary_row(p) for p in profiles]
    df = pd.DataFrame(rows)
    st.markdown("#### Comparison summary")
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_overlaid_year_lines(profiles):
    records = []
    for p in profiles:
        pubs_per_year = p.get("PublicationsPerYear") or {}
        institution = p.get("Institution", "—")
        for year, count in pubs_per_year.items():
            try:
                records.append({
                    "Year": int(year),
                    "Publications": int(count),
                    "Institution": institution,
                })
            except (TypeError, ValueError):
                continue
    if not records:
        return
    df = pd.DataFrame(records).sort_values(["Institution", "Year"])
    st.markdown("#### Publications per year (compared)")
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("Publications:Q", title="Publications",
                axis=alt.Axis(format="d", tickMinStep=1)),
        color=alt.Color("Institution:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Institution", "Year", "Publications"],
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)


def render_top_topics(profile, top_n=10):
    topics = profile.get("TopicDistribution") or []
    if not topics:
        return
    df = pd.DataFrame(topics[:top_n], columns=["Topic", "Count"])
    st.markdown("###### Top Topics")
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Topic:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=300)),
        x=alt.X("Count:Q", title="Papers",
                axis=alt.Axis(format="d", tickMinStep=1)),
        tooltip=["Topic", "Count"],
    ).properties(height=max(180, 24 * len(df)))
    st.altair_chart(chart, use_container_width=True)


def render_top_cited(profile, top_n=5):
    top_papers = profile.get("Top Cited Papers") or []
    if not top_papers:
        return
    rows = []
    for p in top_papers[:top_n]:
        title = (p.get("Title") or "").strip()
        rows.append({
            "Title": title[:90] + ("…" if len(title) > 90 else ""),
            "Citations": int(p.get("Citations") or 0),
            "DOI": p.get("DOI") or "",
        })
    df = pd.DataFrame(rows)
    st.markdown("###### Top-cited papers")
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Title:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=400)),
        x=alt.X("Citations:Q",
                axis=alt.Axis(format="d", tickMinStep=1)),
        tooltip=["Title", "Citations", "DOI"],
    ).properties(height=max(150, 30 * len(df)))
    st.altair_chart(chart, use_container_width=True)


def render_authors_heatmap(profile):
    rows = profile.get("ResearchersWithPublicationsPerYear") or []
    if not rows:
        return
    df = pd.DataFrame(rows)
    if df.empty or "Year" not in df or "Author" not in df:
        return
    df["Year"] = df["Year"].astype(int)
    all_years = list(range(df["Year"].min(), df["Year"].max() + 1))
    # Sort authors by total publications so the most prolific are at the top
    # and the long tail of single-paper co-authors falls toward the bottom.
    author_totals = (
        df.groupby("Author")["Publications"].sum().sort_values(ascending=False)
    )
    all_authors = author_totals.index.tolist()

    full = pd.DataFrame(list(product(all_years, all_authors)), columns=["Year", "Author"])
    df = pd.merge(full, df, on=["Year", "Author"], how="left")
    df["Publications"] = df["Publications"].fillna(0).astype(int)
    df["Titles"] = df.get("Titles", "").fillna("(no publications)")
    df["Year"] = df["Year"].astype(str)

    st.markdown(f"###### Authors × Years ({len(all_authors)} authors)")
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y(
                "Author:N",
                title=None,
                axis=alt.Axis(labelLimit=240),
                sort=all_authors,  # keep prolific-first order
            ),
            color=alt.Color("Publications:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("Author:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Publications:Q"),
                alt.Tooltip("Titles:N", title="Papers"),
            ],
        )
        .properties(height=max(200, 18 * len(all_authors)))
    )
    st.altair_chart(chart, use_container_width=True)


def render_citation_histogram(profile):
    cites = profile.get("CitationDistribution") or []
    if not cites:
        return
    df = pd.DataFrame({"Citations": cites})
    st.markdown("###### Citation distribution")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Citations:Q", bin=alt.Bin(maxbins=20),
                title="Citations per paper"),
        y=alt.Y("count():Q", title="# papers",
                axis=alt.Axis(format="d", tickMinStep=1)),
        tooltip=[alt.Tooltip("count():Q", title="Papers")],
    ).properties(height=200)
    st.altair_chart(chart, use_container_width=True)


def _dot_escape(s):
    return (s or "").replace('\\', '\\\\').replace('"', '\\"')


def render_collab_network(profile, top_n=10):
    collabs = profile.get("Collaborator_Universities") or []
    if not collabs:
        return
    focal = profile.get("Institution", "Institution")
    top = collabs[:top_n]
    if not top:
        return

    counts = []
    for _, c in top:
        try:
            counts.append(int(c))
        except (TypeError, ValueError):
            counts.append(1)
    max_count = max(counts) if counts else 1

    # Undirected (collaboration is symmetric) + circo engine = focal at the
    # center with collaborators arranged in a ring around it. Counts go
    # INSIDE the collaborator node label so they sit on the node's
    # filled background — putting them on edges put them on the canvas
    # where they were unreadable in Streamlit dark mode (black on black).
    lines = [
        "graph G {",
        "  layout=circo;",
        "  overlap=false;",
        "  splines=true;",
        "  bgcolor=transparent;",
        '  size="9,3.5";',
        "  ratio=fill;",
        '  node [style="filled,rounded", shape=box, '
            'fontname="Helvetica", fontsize=10, color="#cccccc", '
            'fontcolor="#1a1a1a", margin="0.12,0.06"];',
        '  edge [color="#9aa0a6"];',
    ]
    lines.append(
        f'  focal [label="{_dot_escape(focal)}", fillcolor="#ffd166", '
        f'fontsize=12, fontcolor="#3b2f00", penwidth=2];'
    )
    for i, ((name, _), count_int) in enumerate(zip(top, counts)):
        nid = f"c{i}"
        edge_weight = max(1.0, min(5.0, 1 + 4 * count_int / max_count))
        node_label = f"{_dot_escape(name)}\\n({count_int} pubs)"
        lines.append(
            f'  {nid} [label="{node_label}", fillcolor="#cfe8ff"];'
        )
        lines.append(
            f'  focal -- {nid} [penwidth={edge_weight:.1f}];'
        )
    lines.append("}")
    dot = "\n".join(lines)

    st.markdown("###### Collaboration network (top 10)")
    try:
        # use_container_width=False lets graphviz honor the size hint above
        # so the chart doesn't fill the entire column.
        st.graphviz_chart(dot, use_container_width=False)
    except Exception as e:
        st.warning(f"Could not render network graph: {e}")


# --- end chart helpers -----------------------------------------------------

with st.sidebar:
    st.title("\U0001F4D8 Description")
    st.markdown("""
This website helps **automatically compare research papers and universities** based on your query and optionally a PDF upload.

It supports answering prompts like:
- “What has the University of Aberdeen contributed to Medical Natural Language Processing?”
- “How does this paper compare to other recent work?”
- “Compare the contributions of TU Wien in the field of efficient information retrieval to the University of Glasgow contributions, over the last 5 years .”

### 🔍 Classification Logic
Each prompt is automatically classified and can answer the following types of questions:

- **`One to many university comparison`**  
  Compare research contributions between one institution and a broader context.  
  _Example:_  
  *“What has the University of Aberdeen contributed to Information Retrieval compared to others?”*

- **`One to one university comparison`**  
  Compare **exactly two** institutions.  
  _Example:_  
  *“What has the University of Aberdeen contributed to Information Retrieval compared to the University of Edinburgh?”*

- **`Single University Overview`**  
  Provide an overview of the research output of one institution.  
  _Example:_  
  *“What has the University of Aberdeen contributed to Information Retrieval over the last 5 years?”*

- **`Comparison of research papers`**  
  Compare a specific paper with other relevant literature.  
  _Requires:* PDF
  _Example:_  
  *“How does the paper compare to other relevant literature <specific criteria>?”*

- **`Question on a single paper`**  
  Answer questions about a **specific paper**.  
  _Example:_  
  *“What is the contribution of this paper?”*

- **`Overview of a domain`**  
  Summarize trends and contributions in a research field **without any specific input document**.  
  _Example:_  
  *“What are the current trends in multimodal learning?”*

---

### 🧠 Data Sources
The app uses different data depending on the classification:

- **OpenAlex**: affiliation + abstracts  
- **CORE UK**: fulltexts  

| Type              | Data Used                      |
|-------------------|-------------------------------|
| `comparison` / `overview` | fulltexts (CORE)         |
| `university`, `single_university`, `university_university` | affiliations + abstracts (OpenAlex) |
| `paper`           | fulltext of provided PDF       |

---

🔁 The system uses caching to avoid recomputation. You can always resubmit without cache if needed.
""")

st.title("Compare Research Papers and Institutions")


def show_recent_queries():
    with st.expander("🕘 Recent Queries (click to expand)", expanded=False):
        try:
            response = requests.get(f"{BACKEND_URL}/api/recent_caches")
            if response.status_code == 200:
                entries = response.json()
                if entries:
                    for i, item in enumerate(entries):
                        col1, col2 = st.columns([6, 1])
                        with col1:
                            st.markdown(f"**{item['prompt'][:80]}...**  \n_Model:_ `{item['model']}`")
                        with col2:
                            if st.button("Use", key=f"use_prompt_{i}"):
                                st.session_state["prompt"] = item["prompt"]
                                st.session_state["model"] = item["model"]
                                st.rerun()
                else:
                    st.info("No recent cache entries found.")
            else:
                st.warning("Could not load recent prompts.")
        except Exception as e:
            st.error(f"Failed to load cache history: {e}")


# --- Stream Backend ---
def stream_backend(prompt, model, pdf_file, no_cache=False, mode="pipeline"):
    files = {"pdf_file": (pdf_file.name, pdf_file, "application/pdf")} if pdf_file else {}
    data = {
        "prompt": prompt,
        "online": "true",
        "model": model,
        "no_cache": str(no_cache).lower(),
        "mode": mode,
    }

    response = requests.post(
        f"{BACKEND_URL}/api/process_input_stream",
        files=files,
        data=data,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    final_result = {}
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            raw = line.removeprefix("data:").strip()
            try:
                parsed = json.loads(raw)
                for key in ["update", "summary", "cached", "profiles", "input", "usage"]:
                    if key in parsed:
                        if key == "update":
                            yield {"type": "update", "message": parsed["update"]}
                        elif key == "cached":
                            final_result["summary"] = parsed["summary"]
                            final_result["cache_timestamp"] = parsed.get("cache_timestamp", "now")
                        else:
                            final_result[key] = parsed[key]
            except json.JSONDecodeError:
                yield {"type": "error", "message": raw}

    if final_result:
        yield {"type": "final", "result": final_result}

# --- UI Elements ---
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
with col1:
    prompt = st.text_area("Enter your prompt", value=st.session_state.get("prompt", ""), height=70)
with col2:
    MODEL_OPTIONS = (
        "gpt-4o-mini",
        "claude-sonnet-4-6",
        "gemini-2.5-pro",
        "qwen-3.5-397b",
        "qwen-3.6-35b",
    )
    model = st.selectbox(
        "Select Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.get("model", "claude-sonnet-4-6")),
    )
with col3:
    MODE_OPTIONS = ("pipeline", "agent")
    mode = st.selectbox(
        "Mode",
        MODE_OPTIONS,
        index=MODE_OPTIONS.index(st.session_state.get("mode", "pipeline")),
        help=(
            "pipeline = deterministic classify→retrieve→summarize flow. "
            "agent = self-refining tool-using agent (slower, more flexible)."
        ),
    )
with col4:
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")


show_recent_queries()

# --- Buttons ---
submit_clicked = st.button("Submit", key="submit_button")
resubmit_clicked = False  # Default

# If last result was from cache, offer resubmit button
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    if result.get("cache_timestamp"):
        with st.container():
            st.warning(f"⚠ Results are cached and may not reflect the latest data. "
                       f"Cache timestamp: {result['cache_timestamp']}")
            if st.button("🔄 Resubmit without cache", key="resubmit_button"):
                st.session_state["resubmit_no_cache"] = True
                st.rerun()

# Handle resubmit state
if st.session_state.get("resubmit_no_cache"):
    no_cache = True
    resubmit_clicked = True
    st.session_state.pop("resubmit_no_cache")
else:
    no_cache = False

# --- Process Streamed Submission ---
if submit_clicked or resubmit_clicked:
    if not prompt:
        st.warning("Please enter a prompt.")
        st.stop()

    st.session_state.pop("last_result", None)
    stream_container = st.empty()
    step_messages = []

    with st.spinner("Processing..."):
        for event in stream_backend(prompt, model, pdf_file, no_cache=no_cache, mode=mode):
            if event["type"] == "update":
                step_messages.append(event["message"])
                with stream_container.container():
                    with st.expander("🛠 Processing Steps", expanded=True):
                        for msg in step_messages:
                            st.markdown(f"🟡 {msg}")
            elif event["type"] == "error":
                st.error(f"🔴 Error parsing message: {event['message']}")
            elif event["type"] == "final":
                st.session_state["last_result"] = event["result"]

    if step_messages:
        with stream_container.container():
            with st.expander("🛠 Processing Steps", expanded=False):
                for msg in step_messages:
                    st.markdown(f"🟢 {msg}")

# --- Show Results ---
if "last_result" in st.session_state:
    result = st.session_state.last_result
    resubmit_clicked = False  # initialize it

    if result.get("cache_timestamp"):
        with st.container():
            st.warning(f"⚠️ Results are cached and may not reflect the latest data. "
                       f"Cache timestamp: {result['cache_timestamp']}")
            if st.button("🔄 Resubmit without cache", key="resubmit_button_clicked"):
                resubmit_clicked = True

    st.markdown("### Response")
    st.write(result["summary"])

    usage = result.get("usage") or {}
    if usage.get("total_tokens"):
        cost = usage.get("cost_usd")
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "n/a"
        st.caption(
            f"🤖 {usage.get('model', '?')} · "
            f"{usage.get('prompt_tokens', 0):,} prompt + "
            f"{usage.get('completion_tokens', 0):,} completion = "
            f"{usage.get('total_tokens', 0):,} tokens · cost: {cost_str}"
        )
    if result.get('profiles'):
        profiles = result['profiles']
        num_profiles = len(profiles)

        # Compute shared y-axis maxes so per-institution bars compare visually
        max_pubs_per_year = 0
        max_collab_pubs = 0
        for profile in profiles:
            pubs_per_year = profile.get("PublicationsPerYear") or {}
            if pubs_per_year:
                max_pubs_per_year = max(max_pubs_per_year, max(pubs_per_year.values()))
            collabs = profile.get("Collaborator_Universities") or []
            if collabs:
                df_c = pd.DataFrame(collabs, columns=["University", "Publications"])
                if not df_c.empty:
                    max_collab_pubs = max(max_collab_pubs, df_c["Publications"].max())

        # Cross-institution overview (only meaningful with 2+ profiles)
        if num_profiles >= 2:
            render_summary_table(profiles)
            render_overlaid_year_lines(profiles)

        cols = st.columns(num_profiles)
        for col, profile in zip(cols, profiles):
            year_start = profile.get("Year Start")
            year_end = profile.get("Year End")
            pubs_per_year = profile.get("PublicationsPerYear") or {}

            if not year_start or not year_end:
                if pubs_per_year:
                    years = sorted(pubs_per_year.keys())
                    year_start = years[0]
                    year_end = years[-1]

            period = f"{year_start} - {year_end}" if year_start and year_end else "N/A"
            name = profile.get("Institution", "—")
            publications = profile.get("Publications", 0)

            with col:
                st.markdown(f"##### {name}")
                st.markdown(f"**Period:** {period}  |  **Publications:** {publications}")

                # Publications per year (per-institution bar)
                st.markdown("###### Publications per Year")
                if year_start and year_end:
                    full_years = list(range(int(year_start), int(year_end) + 1))
                else:
                    full_years = []
                df_y = pd.DataFrame(list(pubs_per_year.items()), columns=["Year", "Publications"])
                if not df_y.empty:
                    df_y["Year"] = df_y["Year"].astype(int)
                    df_full = pd.DataFrame({"Year": full_years}) if full_years else df_y[["Year"]]
                    df_y = pd.merge(df_full, df_y, on="Year", how="left")
                    df_y["Publications"] = df_y["Publications"].fillna(0).astype(int)
                    df_y["Year"] = df_y["Year"].astype(str)
                    chart = alt.Chart(df_y).mark_bar().encode(
                        x=alt.X("Year:O", title="Year"),
                        y=alt.Y("Publications:Q", title="Number of Publications",
                                axis=alt.Axis(format="d", tickMinStep=1),
                                scale=alt.Scale(domain=[0, max_pubs_per_year])),
                        tooltip=["Year", "Publications"],
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

                render_top_topics(profile)
                render_top_cited(profile)
                render_authors_heatmap(profile)
                render_citation_histogram(profile)

                # Collaborator universities (per-institution bar)
                collaborators = profile.get("Collaborator_Universities") or []
                if collaborators:
                    st.markdown("###### Collaborator Universities")
                    df_collab = pd.DataFrame(collaborators, columns=["University", "Publications"])
                    df_collab = df_collab.sort_values("Publications", ascending=False)
                    chart = alt.Chart(df_collab).mark_bar().encode(
                        x=alt.X("University:N", sort=df_collab["University"].tolist(),
                                title="Institution",
                                axis=alt.Axis(labelAngle=-75, labelOverlap=False)),
                        y=alt.Y("Publications:Q", title="Number of Publications",
                                scale=alt.Scale(domain=[0, max_collab_pubs]),
                                axis=alt.Axis(format="d", tickMinStep=1)),
                        tooltip=["University", "Publications"],
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

                render_collab_network(profile)

    # --- Feedback Form ---
    with st.form(key=f"feedback_form"):
        st.markdown("### 💬 Feedback")
        rating = st.radio(
            "How useful was this response?",
            [1, 2, 3, 4, 5],
            format_func=lambda
                x: f"{x} - {'Very Poor' if x == 1 else 'Poor' if x == 2 else 'Neutral' if x == 3 else 'Good' if x == 4 else 'Excellent'}",
            horizontal=True
        )
        comments = st.text_area("Any additional comments or suggestions?")
        submit_feedback = st.form_submit_button("Submit Feedback")

        if submit_feedback:
            st.success("Thank you for your feedback!")

            # Optionally log or send feedback somewhere
            feedback_payload = {
                "prompt": prompt,
                "model": model,
                "file_hash": result["input"].get("file_hash"),
                "rating": rating,
                "comments": comments
            }

            try:
                feedback_response = requests.post(f"{BACKEND_URL}/api/feedback", json=feedback_payload)
                if feedback_response.status_code == 200:
                    st.success("Thank you! Your feedback was saved.")
                else:
                    st.error(f"Feedback error: {feedback_response.status_code} - {feedback_response.text}")
            except Exception as e:
                st.error(f"Failed to send feedback: {e}")
