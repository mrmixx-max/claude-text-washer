import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")  # for `scripts.` package imports

mods = []
try:
    import cli_utils
    mods.append("cli_utils")
except Exception as e:
    print("ERR cli_utils:", e)
try:
    import multi_agent_washer
    mods.append("multi_agent_washer")
except Exception as e:
    print("ERR multi_agent_washer:", e)
try:
    import menu
    mods.append("menu")
except Exception as e:
    print("ERR menu:", e)
try:
    import washer
    mods.append("washer")
except Exception as e:
    print("ERR washer:", e)
try:
    import pipeline
    mods.append("pipeline")
except Exception as e:
    print("ERR pipeline:", e)
try:
    import file_washer
    mods.append("file_washer")
except Exception as e:
    print("ERR file_washer:", e)
try:
    import stat_engine
    mods.append("stat_engine")
except Exception as e:
    print("ERR stat_engine:", e)
try:
    import stat_prompt
    mods.append("stat_prompt")
except Exception as e:
    print("ERR stat_prompt:", e)
try:
    import editor
    mods.append("editor")
except Exception as e:
    print("ERR editor:", e)

print("OK:", mods)
# quick functional checks
from multi_agent_washer import WashCandidate, rank_candidates, format_candidate_summary, format_results_table, AGENT_MODELS
c1 = WashCandidate("a", "llama3.2", "standard", 1.0, 40.0)
c2 = WashCandidate("b", "eurollm-9b", "premium", 2.0, 20.0)
c3 = WashCandidate("c", "lfm25-tool", "fast", 0.5, 60.0)
print("rank:", rank_candidates([c1, c2, c3]).model)
print("summary:", format_candidate_summary(c2))
print("table:")
print(format_results_table([c1, c2, c3], c2))

from editor import TextBuffer, StatusInfo, build_status_info, classify_key, render_fullscreen_preview, render_viewport
buf = TextBuffer.from_text("Hello world. This is a test of the editor.\nSecond line here.")
st = build_status_info(buf, 80)
print("status:", st.render()[:80])
print("stats:", buf.stats())
print("preview lines:", render_fullscreen_preview(buf, 40, 10))
print("classify:", classify_key("\x13"), classify_key("\x10"), classify_key("\x1b[A"))
print("ALL SMOKE OK")
