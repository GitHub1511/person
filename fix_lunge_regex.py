with open(r'C:\Users\Shivi\Downloads\person\embodied_human\skills.py', 'r') as f:
    content = f.read()

# Replace the lunge-press section - use regex-like approach
import re

pattern = r'(if self\._rec_lunged:\n\s+# Landed a lunge at the rock peak: front quad presses straight\n\s+# to standing \(strongest motion available: 200 Nm knee\)\.\n\s+# Coordinate whole body: extend front knee, extend hips, arch spine,\n\s+# press hands - all together for max vertical force\.\n\s+F = "l" if self\._recover_attempts % 2 == 0 else "r"\n\s+B = "r" if F == "l" else "l"\n\s+yield from self\._rec_hold\(\n\s+\{f"knee_{F}": 0\.10, f"hip_{F}_flex": -0\.05,\n\s+f"hip_\{\'r\' if F==\'l\' else \'l\'\}_flex": -0\.05,\n\s+"spine_bend": -0\.25, "chest_bend": -0\.10,\n\s+"sh_l_flex": 0\.20, "sh_r_flex": 0\.20,\n\s+"elbow_l": -0\.30, "elbow_r": -0\.30\},\n\s+10\.0, lambda: self\._rec_com\(\) > 0\.65, "lunge-press"\)\n\s+self\._rec_pressed = True\n\s+self\.events\.append\("stand_up: pressed up from lunge"\)\n\s+elif table_ok:)'

replacement = '''            if self._rec_lunged:
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
            elif table_ok:'''

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content != content:
    with open(r'C:\Users\Shivi\Downloads\person\embodied_human\skills.py', 'w') as f:
        f.write(new_content)
    print('Replaced successfully via regex')
else:
    print('Pattern not found')