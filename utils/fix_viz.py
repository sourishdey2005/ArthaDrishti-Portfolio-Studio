# Read the file
with open('visualization.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all problematic lines by removing the fill parameters - handle any rgba value
import re
pattern = r", fill='[a-z]+', fillcolor='rgba\([0-9., ]+\)'"
content = re.sub(pattern, '', content)

# Write the cleaned file
with open('visualization.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all!")
