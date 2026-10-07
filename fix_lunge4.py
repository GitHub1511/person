with open(r'C:\Users\Shivi\Downloads\person\embodied_human\skills.py', 'r') as f:
    content = f.read()

old = """            if self._rec_lunged:
                # Landed a lunge at the rock peak: front quad presses straight
                # to standing (strongest motion available: 200 Nm knee).
                # Coordinate whole body: extend front knee, extend hips, arch spine,
                # press hands - all together for max vertical force.
                F = "l" if self._recover_attempts % 2 == 0 else "r"
                B = "r" if F == "l" else "l"
                yield from self._rec_hold(
                    {f"knee_{F}": 0.10, f"hip_{F}_flex": -0.05,
                     f"hip_{\'r\' if F==\'l\' else \'l\'}": -0.05,
                     "spine_bend": -0.25, "chest_bend": -0.10,
                     "sh_l_flex": 0.20, "sh_r_flex": 0.20,
                     "elbow_l": -0.30, "elbow_r": -0.30},
                    10.0, lambda: self._rec_com() > 0.65, "lunge-press")
                self._rec_pressed = True
                self.events.append("stand_up: pressed up from lunge")
            elif table_ok:"""

new = """            if self._rec_lunged:
                # Landed a lunge at the rock peak: full push-up press
                # combining cobra arch, hip extension, and front knee drive.
                F = "l" if self._recover_attempts % 2 == 0 else "r"
                B = "r" if F == "l" else "l"
                yield from self._rec_hold(
                    {f"knee_{F}": 0.08, f"hip_{F}_flex": -0.02,
                     f"hip_{B}_flex": -0.02, f"knee_{B}": 0.10,
                     "spine_bend": -0.35, "chest_bend": -0.15,
                     "sh_l_flex": 0.25, "sh_r_flex": 0.25,
                     "elbow_l": -0.20, "elbow_r": -0.20},
                    20.0, lambda: self._rec_com() > 0.70, "lunge-press")
                self._rec_pressed = True
                self.events.append("stand_up: pressed up from lunge")
            elif table_ok:"""

if old in content:
    content = content.replace(old, new)
    with open(r'C:\Users\Shivi\Downloads\person\embodied_human\skills.py', 'w') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('OLD STRING NOT FOUND')
    idx = content.find('if self._rec_lunged:')
    if idx >= 0:
        print('Found at', idx)
        print(repr(content[idx:idx+1000]))