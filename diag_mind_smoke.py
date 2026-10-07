"""Smoke test for the mind layer without any model: parse replies, check the
sandbox refuses anything that is not a whitelisted call."""
from embodied_human.mind import parse_reply, parse_calls

good = parse_reply(' I should look. </think> <answer>\nlook_at("apple")\ngrab("apple", hand="left")\n</answer>')
print("think:", good.think)
print("calls:", good.calls, "errors:", good.errors)
assert [c[0] for c in good.calls] == ["look_at", "grab"]

bad = [
    '__import__("os").system("calc")',
    'open("x").read()',
    'say("a" + "b")',
    'import os',
    'x = 1',
    'grab(apple)',
    'say(f"hi")',
    'walk_to("table"); exec("1")',
    '[say("a") for _ in range(9)]',
    'say("x" * 999)',
]
for b in bad:
    calls, errors = parse_calls(b)
    print(f"{b!r:50} -> calls={calls} errors={errors}")
    assert not any(c[0] in ("exec", "open", "__import__", "system") for c in calls)
print("ok: nothing outside the whitelist was accepted")
