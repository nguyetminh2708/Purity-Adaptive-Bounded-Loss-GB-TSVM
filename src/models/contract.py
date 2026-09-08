"""
Input-shape contract — the original code's conventions, derived EMPIRICALLY.

The table below was not inferred by reading the code. Each class was actually
called with X=(k,d) and X=(k,d+1), and whichever did not blow up on shape was
recorded. This is the root cause of bug L1: GridSearchCV hands the same X to
both fit and score, but the classes disagree on how many columns X must have.

    class                   predict takes   returns
    ----------------------- --------------- ---------------------------
    GBTSVM                  (k, d+1)        (1, k)   <- transposed
    PinGBTSVM               (k, d+1)        (1, k)   <- transposed
    LGBTSVM                 (k, d+1)        (k, k)   <- L5, bad broadcast
    OvR_LGBTSVM             (k, d+1)        (k,)
    OvO_GBTSVM              (k, d)          (k,)
    MultiLabelGBTSVM        (k, d)          (k,)
    OvO_PinGBTSVM           (k, d)          (k,)
    OvO_LGBTSVM             either          (k,)
    OVR_GBTSVM              (k, d)          (k, n_classes) <- multi-label matrix
    OVR_PinGBTSVM           (k, d)          (k, n_classes) <- multi-label matrix

L5 (LGBTSVM):  w1 has shape (d,1) while b1*ones(m) has shape (m,), so
    np.dot(X, w1) + b1*ones(m)  ->  (m, 1) + (m,)  ->  (m, m)
The DIAGONAL of that matrix holds the correct values, but score() reads ROW 0.
Every LGBTSVM number in the old results therefore comes from a wrong decision
function.

This module exists only to REPLAY the old path for side-by-side comparison.
The fixed path lives in models.py.
"""

import io
import contextlib

import numpy as np

# Does this class's predict need an extra trailing column (dummy label/radius)?
NEEDS_TRAILING_COL = {
    "GBTSVM": True,
    "PinGBTSVM": True,
    "LGBTSVM": True,
    "OvR_LGBTSVM": True,
    "OvO_GBTSVM": False,
    "MultiLabelGBTSVM": False,
    "OvO_PinGBTSVM": False,
    "OvO_LGBTSVM": False,
    "OVR_GBTSVM": False,
    "OVR_PinGBTSVM": False,
}

# Output shapes that need special handling
OUT_TRANSPOSED = {"GBTSVM", "PinGBTSVM"}           # (1, k)
OUT_BROADCAST_BUG = {"LGBTSVM"}                     # (k, k) — take row 0, as score() does
OUT_MULTILABEL = {"OVR_GBTSVM", "OVR_PinGBTSVM"}    # (k, n_classes)


def predict_original(name, model, X, classes=None):
    """
    Call the original class's predict and normalise the output to (k,).

    Reproduces the old behaviour EXACTLY, bugs included, so that an
    "old vs new" table is comparable. For OVR_*, the multi-label matrix is
    reduced to labels by column argmax — the only place a choice is forced,
    because the original formulation never produces labels at all.
    """
    Z = np.column_stack([X, np.zeros(len(X))]) if NEEDS_TRAILING_COL[name] else X

    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(model.predict(Z))

    if name in OUT_BROADCAST_BUG:
        return out[0]                     # the original score() reads row 0
    if name in OUT_TRANSPOSED:
        return out.ravel()
    if name in OUT_MULTILABEL:
        if classes is None:
            raise ValueError(f"{name} needs `classes` to reduce the matrix to labels")
        return np.asarray(classes)[np.asarray(out).argmax(axis=1)]
    return out.ravel()
