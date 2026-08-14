# Reflection

With more data, I'd pull real historical tickets instead of templated synthetic
ones — the current dataset scores 100% on its held-out test set mainly because
the test tickets share sentence structure with the training tickets, not
because the model is truly that strong (the "vacation days" example in
train_model.py, worded differently from the HR templates, gets misclassified
and drops to ~29% confidence, which is exactly what real unseen phrasing would
do). With more time I'd add a "None of the above / Other" fallback category
for genuinely ambiguous tickets, weight recent tickets more heavily so the
model adapts as language changes, and track live accuracy against
human-corrected labels to catch drift instead of trusting one static
evaluation.
