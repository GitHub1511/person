with open(r'C:\Users\Shivi\Downloads\person\embodied_human\skills.py', 'r') as f:
    lines = f.readlines()

# Find the exact line with "if self._rec_lunged:"
for i, line in enumerate(lines):
    if 'if self._rec_lunged:' in line and 'Landed a lunge' in lines[i+1]:
        start_idx = i
        print(f'Found at line {i+1}')
        break

# Find the "elif table_ok:" after this block
for i in range(start_idx, len(lines)):
    if 'elif table_ok:' in lines[i] and i > start_idx:
        end_idx = i
        print(f'Block ends at line {i+1}')
        break

print(f'Replacing lines {start_idx+1} to {end_idx}')

# Build new lines
new_lines = [
    '            if self._rec_lunged:\n',
    '                # Landed a lunge at the rock peak: full push-up press\n',
    '                # combining cobra arch, hip extension, and front knee drive.\n',
    '                F = "l" if self._recover_attempts % 2 == 0 else "r"\n',
    '                B = "r" if F == "l" else "l"\n',
    '                yield from self._rec_hold(\n',
    '                    {f"knee_{F}": 0.08, f"hip_{F}_flex": -0.02,\n',
    '                     f"hip_{B}_flex": -0.02, f"knee_{B}": 0.10,\n',
    '                     "spine_bend": -0.35, "chest_bend": -0.15,\n',
    '                     "sh_l_flex": 0.25, "sh_r_flex": 0.25,\n',
    '                     "elbow_l": -0.20, "elbow_r": -0.20},\n',
    '                    20.0, lambda: self._rec_com() > 0.70, "lunge-press")\n',
    '                self._rec_pressed = True\n',
    '                self.events.append("stand_up: pressed up from lunge")\n',
    '            elif table_ok:\n'
]

# Replace
lines[start_idx:end_idx] = new_lines

with open(r'C:\Users\Shivi\Downloads\person\embodied_human\skills.py', 'w') as f:
    f.writelines(lines)

print('Replaced successfully')