"""
Seeing the person: rendering, speech bubbles, thought bubbles, and the window.

Everything the person says appears above their head in a white speech bubble;
everything they think appears above that in grey italics.  MuJoCo's own
viewer can only draw one fixed-colour bitmap font, so the scene is rendered
off-screen and the text is composited on top with PIL, with each bubble
anchored to the head's position projected through the camera.

``SceneRenderer`` works headless (it is how the test frames in ``out/`` were
made); ``App`` wraps it in a Tk window with a log, a box to talk to the person,
and camera controls.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required: python -m pip install mujoco") from exc

from PIL import Image, ImageDraw, ImageFont

_FONT_DIRS = ["C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/",
              "/Library/Fonts/", "/System/Library/Fonts/Supplemental/"]


def _font(names: list[str], size: int):
    for d in _FONT_DIRS:
        for n in names:
            try:
                return ImageFont.truetype(d + n, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


REGULAR = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf"]
ITALIC = ["segoeuii.ttf", "ariali.ttf", "DejaVuSans-Oblique.ttf", "Arial Italic.ttf"]
BOLD = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class SceneRenderer:
    def __init__(self, agent, size=(960, 600), mind=None):
        self.agent = agent
        self.mind = mind
        self.W, self.H = size
        self.renderer = mujoco.Renderer(agent.model, self.H, self.W)
        self.ego = mujoco.Renderer(agent.model, 120, 160)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.azimuth, self.elevation, self.distance = 148.0, -12.0, 3.2
        self.follow = True
        self.lookat = np.array([0.0, -0.3, 0.9])
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.show_ego = True
        self.f_bold = _font(BOLD, 14)
        self.f_hud = _font(REGULAR, 14)
        self._font_cache: dict = {}

    # ------------------------------------------------------------------
    def set_view(self, name: str) -> None:
        views = {"front": (180.0 - 32.0, -10.0, 3.0), "side": (90.0 + 180, -8.0, 3.2),
                 "three_quarter": (148.0, -12.0, 3.4), "close": (150.0, -8.0, 1.8),
                 "top": (150.0, -60.0, 4.5), "back": (0.0 + 20, -10.0, 3.4)}
        if name in views:
            self.azimuth, self.elevation, self.distance = views[name]

    def _fonts(self, size: int):
        key = size
        if key not in self._font_cache:
            self._font_cache[key] = (_font(REGULAR, size), _font(ITALIC, size))
        return self._font_cache[key]

    # ------------------------------------------------------------------
    def render(self, overlay: bool = True) -> Image.Image:
        ag = self.agent
        if self.follow and ag.state is not None:
            c = ag.state.com
            target = np.array([c[0], c[1], 0.80])
            self.lookat += 0.15 * (target - self.lookat)
        self.cam.lookat[:] = self.lookat
        self.cam.azimuth = self.azimuth
        self.cam.elevation = self.elevation
        self.cam.distance = self.distance
        self.renderer.update_scene(ag.data, self.cam, self.opt)
        img = Image.fromarray(self.renderer.render().copy())
        if overlay:
            self._overlay(img)
            if self.show_ego:
                # the inset costs as much as the main view: refresh it at ~1/4 rate
                self._ego_n = getattr(self, "_ego_n", 0) + 1
                if self._ego_n % 4 == 1 or getattr(self, "_ego_img", None) is None:
                    self.ego.update_scene(ag.data, "egocentric", self.opt)
                    self._ego_img = Image.fromarray(self.ego.render().copy())
                e = self._ego_img
                img.paste(e, (self.W - 160 - 8, 8))
                ImageDraw.Draw(img).rectangle([self.W - 168, 8, self.W - 8, 128],
                                              outline=(200, 200, 200))
        return img

    # ------------------------------------------------------------------
    def project(self, p: np.ndarray) -> tuple[float, float, float] | None:
        cam = self.renderer.scene.camera[0]
        pos = np.array(cam.pos)
        f = np.array(cam.forward)
        u = np.array(cam.up)
        r = np.cross(f, u)
        v = np.asarray(p) - pos
        z = float(np.dot(v, f))
        if z <= 0.05:
            return None
        fovy = math.radians(self.agent.model.vis.global_.fovy)
        th = math.tan(fovy / 2)
        aspect = self.W / self.H
        x = float(np.dot(v, r)) / z / (th * aspect)
        y = float(np.dot(v, u)) / z / th
        px = self.W / 2 * (1 + x)
        py = self.H / 2 * (1 - y)
        scale = (self.H / 2) / (th * z)          # pixels per metre at that depth
        return px, py, scale

    # ------------------------------------------------------------------
    def _overlay(self, img: Image.Image) -> None:
        ag = self.agent
        mind = self.mind
        head = ag.skills.world.head_pos() + np.array([0.0, 0.0, 0.13])
        pr = self.project(head)
        draw = ImageDraw.Draw(img, "RGBA")
        if pr is not None:
            px, py, scale = pr
            size = int(np.clip(0.020 * scale + 5, 13, 26))
            f_reg, f_it = self._fonts(size)
            y_cursor = py - 10
            # ---- speech: above the head ---------------------------------
            text, alpha = ag.skills.speech.visible_text()
            if text:
                y_cursor = self._bubble(draw, px, y_cursor, text, f_bold=_font(BOLD, size),
                                        fill=(255, 255, 255, int(235 * alpha)),
                                        ink=(20, 20, 24, int(255 * alpha)), tail=True,
                                        outline=(40, 40, 50, int(255 * alpha)), size=size)
            # ---- thoughts: above the speech, grey and italic -------------
            thought, a_t = self._current_thought()
            if thought and a_t > 0.02:
                self._bubble(draw, px, y_cursor - 6, thought, f_bold=f_it,
                             fill=(70, 72, 78, int(150 * a_t)),
                             ink=(190, 192, 198, int(255 * a_t)), tail=False,
                             outline=(120, 122, 130, int(170 * a_t)), size=size - 2,
                             cloud=True, italic=True)
        self._hud(draw)

    def _current_thought(self) -> tuple[str, float]:
        mind = self.mind
        if mind is None:
            return "", 0.0
        if mind.thinking and mind.live_thought:
            return mind.live_thought, 1.0
        if mind.thought:
            age = self.agent.t - mind.thought_t
            hold = 9.0 + 0.04 * len(mind.thought)
            if age < hold:
                return mind.thought, 1.0
            if age < hold + 3.0:
                return mind.thought, 1.0 - (age - hold) / 3.0
        return "", 0.0

    def _bubble(self, draw, cx, bottom, text, f_bold, fill, ink, tail, outline, size,
                cloud=False, italic=False) -> float:
        """Draw a bubble whose bottom edge is at ``bottom``; return its top."""
        max_w = int(np.clip(self.W * 0.34, 220, 380))
        font = f_bold
        if cloud and len(text) > 260:
            text = "…" + text[-258:]
        lines = wrap_text(draw, text, font, max_w - 18)
        if cloud:
            lines = lines[-6:]
        lh = size + 4
        w = int(max((draw.textlength(l, font=font) for l in lines), default=20)) + 20
        h = lh * len(lines) + 14
        x0 = int(np.clip(cx - w / 2, 6, self.W - w - 6))
        y1 = int(bottom) - (8 if tail else 0)
        y0 = y1 - h
        if y0 < 4:
            y0, y1 = 4, 4 + h
        draw.rounded_rectangle([x0, y0, x0 + w, y1], radius=12 if not cloud else 18,
                               fill=fill, outline=outline, width=1)
        if tail:
            tx = int(np.clip(cx, x0 + 14, x0 + w - 14))
            draw.polygon([(tx - 7, y1 - 1), (tx + 7, y1 - 1), (tx, y1 + 9)], fill=fill)
        if cloud:
            # little thought dots leading down to the head
            for i, r in enumerate((4, 3, 2)):
                draw.ellipse([cx - r + 3 * i, y1 + 4 + 8 * i, cx + r + 3 * i, y1 + 4 + 8 * i + 2 * r],
                             fill=fill, outline=outline)
        for i, l in enumerate(lines):
            draw.text((x0 + 10, y0 + 7 + i * lh), l, font=font, fill=ink)
        return y0

    def _hud(self, draw) -> None:
        ag = self.agent
        mind = self.mind
        sk = ag.skills
        L = [f"t = {ag.t:6.1f} s"]
        g = ag.gait
        L.append(f"gait: {g.diag.mode if g.walking else 'standing'}  steps {g.steps}")
        L.append("doing: " + sk.current_description())
        h = [f"{'L' if s == 'l' else 'R'}: {sk.held[s]}" for s in "lr" if sk.held[s]]
        L.append("holding: " + (", ".join(h) if h else "nothing"))
        if mind is not None:
            tag = "" if mind.backend.is_language_model else "  [NOT A LANGUAGE MODEL]"
            L.append(f"mind: {mind.backend.name}{tag}")
            L.append(f"mind status: {mind.status}")
        a = ag.affect_frame
        if a is not None:
            L.append(f"feels: {a.dominant_emotion}  v {a.valence:+.2f}  a {a.arousal:.2f}")
        pad = 6
        wmax = max(draw.textlength(l, font=self.f_hud) for l in L) + 2 * pad
        top = self.H - 10 - 18 * len(L)          # bottom-left, out of the bubbles' way
        draw.rectangle([4, top - 4, 4 + wmax, self.H - 4], fill=(10, 12, 16, 150))
        for i, l in enumerate(L):
            col = (255, 150, 130, 255) if "NOT A LANGUAGE" in l else (225, 230, 235, 255)
            draw.text((4 + pad, top + 18 * i), l, font=self.f_hud, fill=col)


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
class App:
    def __init__(self, agent, mind, *, size=(960, 600), speed=1.0, voice=False):
        import tkinter as tk
        from tkinter import scrolledtext
        from PIL import ImageTk
        self.tk = tk
        self.ImageTk = ImageTk
        self.agent = agent
        self.mind = mind
        self.root = tk.Tk()
        self.root.title("Embodied human")
        self.view = SceneRenderer(agent, size, mind)
        self.speed = tk.DoubleVar(value=speed)
        self.paused = tk.BooleanVar(value=False)
        self.follow = tk.BooleanVar(value=True)
        self.ego = tk.BooleanVar(value=True)

        left = tk.Frame(self.root)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Label(left, bd=0)
        self.canvas.pack()
        bar = tk.Frame(left)
        bar.pack(fill="x")
        for name in ("three_quarter", "front", "side", "back", "close", "top"):
            tk.Button(bar, text=name.replace("_", " "), width=11,
                      command=lambda n=name: self.view.set_view(n)).pack(side="left")
        tk.Checkbutton(bar, text="follow", variable=self.follow).pack(side="left")
        tk.Checkbutton(bar, text="eye view", variable=self.ego).pack(side="left")
        tk.Checkbutton(bar, text="pause", variable=self.paused).pack(side="left")
        tk.Label(bar, text="speed").pack(side="left")
        tk.Scale(bar, from_=0.25, to=4.0, resolution=0.25, orient="horizontal",
                 variable=self.speed, length=110).pack(side="left")
        self.rt_label = tk.Label(bar, text="", width=14)
        self.rt_label.pack(side="left")

        talk = tk.Frame(left)
        talk.pack(fill="x", pady=4)
        tk.Label(talk, text="say to the person:").pack(side="left")
        self.entry = tk.Entry(talk)
        self.entry.pack(side="left", fill="x", expand=True, padx=4)
        self.entry.bind("<Return>", lambda e: self.send())
        tk.Button(talk, text="say", command=self.send).pack(side="left")

        right = tk.Frame(self.root)
        right.pack(side="right", fill="both")
        tk.Label(right, text="inner life", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.log = scrolledtext.ScrolledText(right, width=52, height=38, wrap="word",
                                             state="disabled", font=("Segoe UI", 10))
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("thought", foreground="#7a7d85", font=("Segoe UI", 10, "italic"))
        self.log.tag_config("speech", foreground="#111111", font=("Segoe UI", 10, "bold"))
        self.log.tag_config("heard", foreground="#1b5e20", font=("Segoe UI", 10, "bold"))
        self.log.tag_config("action", foreground="#0b57a4")
        self.log.tag_config("event", foreground="#8a5a00")
        self.log.tag_config("sys", foreground="#aa3333")

        self.view.canvas_dirty = True
        self._last_render = 0.0
        self._wall0 = time.perf_counter()
        self._sim0 = agent.t
        self._seen = {"thought": 0, "speech": 0, "event": 0}
        self._speech_logged = 0
        self._last_action = ""
        self._drag = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<ButtonPress-3>", self._press)
        self.canvas.bind("<B3-Motion>", self._motion_pan)
        self._append("sys", f"mind backend: {mind.backend.name}")
        if not mind.backend.is_language_model:
            self._append("sys", "NOTE: this is the scripted stand-in, not AZR. See README.")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._running = True
        self.root.after(5, self._tick)

    # ------------------------------------------------------------------
    def send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.agent.skills.hear(text)
        self._append("heard", f'you: "{text}"')
        self.mind.poke()

    def _append(self, tag: str, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _press(self, e):
        self._drag = (e.x, e.y)

    def _motion(self, e):
        if self._drag:
            dx, dy = e.x - self._drag[0], e.y - self._drag[1]
            self.view.azimuth -= 0.4 * dx
            self.view.elevation = float(np.clip(self.view.elevation - 0.3 * dy, -85, 10))
            self._drag = (e.x, e.y)

    def _motion_pan(self, e):
        self.follow.set(False)
        if self._drag:
            dx, dy = e.x - self._drag[0], e.y - self._drag[1]
            az = math.radians(self.view.azimuth)
            self.view.lookat[0] += 0.004 * (dx * math.sin(az))
            self.view.lookat[1] -= 0.004 * (dx * math.cos(az))
            self.view.lookat[2] = float(np.clip(self.view.lookat[2] + 0.004 * dy, 0.1, 2.0))
            self._drag = (e.x, e.y)

    def _wheel(self, e):
        self.view.distance = float(np.clip(self.view.distance * (0.9 if e.delta > 0 else 1.1),
                                           0.8, 12.0))

    # ------------------------------------------------------------------
    def _sync_log(self) -> None:
        m = self.mind
        for t in m.transcript[self._seen["thought"]:]:
            if t["think"]:
                self._append("thought", t["think"])
            if t["answer"].strip():
                self._append("action", "→ " + " ; ".join(
                    s.strip() for s in t["answer"].strip().splitlines() if s.strip()))
            for er in t["errors"]:
                self._append("sys", f"(answer problem: {er})")
        self._seen["thought"] = len(m.transcript)
        hist = self.agent.skills.speech.history
        for ts, text in hist[self._speech_logged:]:
            self._append("speech", f'"{text}"')
        self._speech_logged = len(hist)
        res = self.agent.skills.last_result
        if res and res != self._last_action:
            self._append("event", res)
            self._last_action = res

    def _tick(self) -> None:
        if not self._running:
            return
        ag = self.agent
        t_wall = time.perf_counter()
        if not self.paused.get():
            target_sim = self._sim0 + (t_wall - self._wall0) * self.speed.get()
            budget_end = t_wall + 0.065
            while ag.t < target_sim and time.perf_counter() < budget_end:
                ag.step()
            if ag.t < target_sim - 0.5:        # can't keep up: don't accumulate debt
                self._sim0 = ag.t - (t_wall - self._wall0) * self.speed.get()
        else:
            self._wall0 = t_wall
            self._sim0 = ag.t
        now = time.perf_counter()
        if now - self._last_render > 0.10:         # ~10 fps: physics is the scarce thing
            self.view.follow = self.follow.get()
            self.view.show_ego = self.ego.get()
            img = self.view.render()
            self._photo = self.ImageTk.PhotoImage(img)
            self.canvas.configure(image=self._photo)
            self._last_render = now
            self._sync_log()
            rt = (ag.t - self._sim0) / max(now - self._wall0, 1e-6) if not self.paused.get() else 0
            self.rt_label.configure(text=f"{ag.t:6.1f}s")
        self.root.after(1, self._tick)

    def close(self) -> None:
        self._running = False
        self.mind.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
