"""
Complete Pipeline to Remove Neutral Stance

This guide walks through removing neutral_stance class and retraining models.

Steps:
1. Remove neutral_stance data folders
2. Update CLASS_NAMES in all Python files
3. Delete old features and models
4. Regenerate features
5. Retrain models

Run each command manually to ensure everything works correctly.
"""

print("""
╔════════════════════════════════════════════════════════════╗
║  REMOVE NEUTRAL STANCE - COMPLETE PIPELINE                 ║
╚════════════════════════════════════════════════════════════╝

STEP 1: Remove neutral_stance data folders
-------------------------------------------
python remove_neutral_stance.py

This deletes:
  - dataset_split/train/*/neutral_stance/
  - dataset_split/test/*/neutral_stance/
  - reference_poses/*/neutral_stance/


STEP 2: Update CLASS_NAMES (13 → 12 classes)
---------------------------------------------
python update_class_names.py

Updates CLASS_NAMES in all Python files to remove 'neutral_stance'


STEP 3: Clean old features and models
--------------------------------------
# Remove old feature templates (13 classes)
rm hybrid_classifier/feature_templates.json

# Remove old features (13 classes)
rm -r hybrid_classifier/hybrid_features_v2/

# Backup old models (optional)
mkdir hybrid_classifier/models/backup_13classes/
cp hybrid_classifier/models/*.pth hybrid_classifier/models/backup_13classes/


STEP 4: Regenerate reference features (12 classes)
---------------------------------------------------
python hybrid_classifier/1_extract_reference_features.py


STEP 5: Regenerate node + hybrid features (12 classes)
-------------------------------------------------------
python hybrid_classifier/2b_generate_node_hybrid_features.py


STEP 6: Retrain all models (12 classes)
----------------------------------------
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --epochs 150

Or train individually:
  python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint front --epochs 150
  python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint left --epochs 150
  python hybrid_classifier/4c_train_hybrid_gcn_v2.py --viewpoint right --epochs 150


STEP 7: Evaluate models (12 classes)
-------------------------------------
python hybrid_classifier/4d_evaluate_model.py --viewpoint front
python hybrid_classifier/4d_evaluate_model.py --viewpoint left
python hybrid_classifier/4d_evaluate_model.py --viewpoint right


STEP 8: Update application (TuroArnis repo)
--------------------------------------------
Update CLASS_NAMES in app repository to 12 classes
Add confidence thresholding logic:

  if max_confidence < 0.7:
      display: "Ready - position yourself"
  else:
      display: technique_name


═══════════════════════════════════════════════════════════

Expected Benefits:
  ✓ Cleaner predictions (12 real techniques)
  ✓ Higher confidence per technique
  ✓ Simpler app logic (no "neutral stance" handling)
  ✓ Better model focus

Expected Training Time:
  ~2-3 hours for all 3 viewpoints (150 epochs each)

═══════════════════════════════════════════════════════════
""")
