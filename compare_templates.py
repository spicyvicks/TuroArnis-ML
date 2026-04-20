import json

# Load old and new templates
with open('hybrid_classifier/feature_templates_backup_20260420.json') as f:
    old = json.load(f)

with open('hybrid_classifier/feature_templates.json') as f:
    new = json.load(f)

print("=== TEMPLATE STATISTICS COMPARISON ===\n")

# Compare crown_thrust_correct
key = 'front_crown_thrust_correct'
if key in old and key in new:
    print(f"{key}:")
    print(f"{'Feature':<30} {'Old Std':<12} {'New Std':<12} {'Change'}")
    print("-" * 70)
    
    angle_features = [k for k in old[key].keys() if 'angle' in k]
    for feat in angle_features:
        old_std = old[key][feat].get('std', 0)
        new_std = new[key][feat].get('std', 0) if feat in new[key] else 0
        
        if old_std > 0:
            change = ((new_std - old_std) / old_std) * 100
            status = "BETTER" if new_std < old_std else "WORSE"
            print(f"{feat:<30} {old_std:>10.1f}  {new_std:>10.1f}  {change:>+6.1f}% {status}")
        else:
            print(f"{feat:<30} {old_std:>10.1f}  {new_std:>10.1f}     N/A")

print("\n=== KEY METRICS ===")
print(f"Old templates total: {len(old)}")
print(f"New templates total: {len(new)} (only front so far)")
