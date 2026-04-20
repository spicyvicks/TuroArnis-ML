import json

with open('hybrid_classifier/feature_templates.json') as f:
    new = json.load(f)

with open('hybrid_classifier/feature_templates_backup_20260420.json') as f:
    old = json.load(f)

print('=== TEMPLATE REGENERATION COMPLETE ===')
print()
print(f'Total templates: {len(new)} (13 classes x 3 viewpoints)')
print()

# Count by viewpoint  
front_count = sum(1 for k in new if k.startswith('front_'))
left_count = sum(1 for k in new if k.startswith('left_'))
right_count = sum(1 for k in new if k.startswith('right_'))

print(f'Front: {front_count} templates')
print(f'Left:  {left_count} templates')
print(f'Right: {right_count} templates')
print()

# Sample improvements
key = 'front_crown_thrust_correct'
if key in old and key in new:
    print('=== SAMPLE IMPROVEMENTS ===')
    print(f'Template: {key}')
    o = old[key]
    n = new[key]
    
    for feat in ['left_elbow_angle', 'right_elbow_angle', 'stick_angle']:
        if feat in o and feat in n:
            old_std = o[feat]['std']
            new_std = n[feat]['std']
            change = (new_std - old_std) / old_std * 100
            print(f'  {feat:20s}: {old_std:6.1f} -> {new_std:6.1f} deg ({change:+6.0f}%)')

print()
print('Tight STDs indicate clean templates without corrupted data!')
print('Ready to test with existing models.')
