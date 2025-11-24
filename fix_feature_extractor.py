import re

with open('/home/cs/DecoupleNet-main/model/feature_extractor.py', 'r') as f:
    content = f.read()

new_content = re.sub(
    r'if pretrained:\s+checkpoint = torch\.load\(path\)',
    'if pretrained:\n        if path.startswith("http"):\n            import torch.hub\n            checkpoint = torch.hub.load_state_dict_from_url(path, progress=True)\n        else:\n            checkpoint = torch.load(path)',
    content
)

with open('/home/cs/DecoupleNet-main/model/feature_extractor.py', 'w') as f:
    f.write(new_content)

print("File modified successfully") 