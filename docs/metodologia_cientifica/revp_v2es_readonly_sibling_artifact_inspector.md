# REV-P v2es readonly sibling artifact inspection

Rows generated: 69

Read-only inspection of sibling worktrees; no copy or restore is performed.

Allowed claim: review-only controlled recovery workflow; restores traceability when valid artifacts exist but does not close operational ground truth, labels, negatives, training, detection, or prediction

Forbidden claim: operational ground truth|binary label|formal negative|supervised dataset|training release|detection claim|prediction claim|automatic human decision

Global state: ground_truth_operational_status=ABSENT; formal_labels_available=ABSENT; formal_negatives_available=ABSENT; training_ready=false; supervised_model_allowed=false; prediction_claim_allowed=false; automatic_detection_claim_allowed=false; operational_validation_claim_allowed=false; negative_by_absence_allowed=false; random_background_negative_allowed=false; decision_locked=false.
