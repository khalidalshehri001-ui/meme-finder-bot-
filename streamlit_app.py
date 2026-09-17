"""Meme Reply Bot - a chat box that answers you with reaction GIFs."""

import streamlit as st

from meme_bot import DEFAULT_MODE, MODES, find_memes

st.set_page_config(page_title="Meme Reply Bot", page_icon="😂")

st.title("😂 Meme Reply Bot")
st.caption("Tell me what happened. I'll reply with memes.")

if "OPENAI_API_KEY" not in st.secrets:
    st.error(
        "No `OPENAI_API_KEY` found. Add it in **Settings → Secrets** "
        "on Streamlit Cloud, or in `.streamlit/secrets.toml` locally."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Meme mode")
    mode = st.radio(
        "How should I reply?",
        list(MODES),
        index=list(MODES).index(DEFAULT_MODE),
        format_func=lambda m: f"{MODES[m]['emoji']}  {m}",
        key="mode",
    )
    st.caption(MODES[mode]["blurb"])

    st.divider()
    st.header("Chat")
    count = len(st.session_state.messages)
    st.caption(f"{count} message{'' if count == 1 else 's'} so far")
    if st.button("🗑️ Clear chat", width="stretch", disabled=not count):
        st.session_state.messages = []
        st.rerun()


def render(msg):
    """Draw one chat message, with its GIFs if it has any."""
    with st.chat_message(msg["role"]):
        if msg.get("mode"):
            cfg = MODES.get(msg["mode"])
            if cfg:
                st.caption(f"{cfg['emoji']} {msg['mode']}")
        st.markdown(msg["content"])
        gifs = msg.get("gifs") or []
        if gifs:
            for col, url in zip(st.columns(len(gifs)), gifs):
                col.image(url, width="stretch")


for msg in st.session_state.messages:
    render(msg)

if prompt := st.chat_input("My boss scheduled a meeting at 5pm on a Friday..."):
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render(user_msg)

    with st.chat_message("assistant"):
        st.caption(f"{MODES[mode]['emoji']} {mode}")
        with st.spinner(f"Finding a {mode.lower()} reaction..."):
            try:
                caption, gifs = find_memes(prompt, how_many=2, mode=mode)
            except Exception as e:
                caption, gifs = f"Something broke: {e}", []

        st.markdown(caption)
        if gifs:
            for col, url in zip(st.columns(len(gifs)), gifs):
                col.image(url, width="stretch")
        else:
            st.caption("(couldn't find a GIF for that one)")

    st.session_state.messages.append(
        {"role": "assistant", "content": caption, "gifs": gifs, "mode": mode}
    )
