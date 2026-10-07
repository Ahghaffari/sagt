import numpy as np
import torch


def _np_mask(y_true, null_val):
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)
    mask = mask.astype("float32")
    mask /= np.mean(mask)
    return np.nan_to_num(mask)


def MAE_np(y_true, y_pred, null_val=0.0):
    mask = _np_mask(y_true, null_val)
    return np.mean(np.nan_to_num(np.abs(y_pred - y_true) * mask))


def RMSE_np(y_true, y_pred, null_val=0.0):
    mask = _np_mask(y_true, null_val)
    return np.sqrt(np.mean(np.nan_to_num(np.square(y_pred - y_true) * mask)))


def MAPE_np(y_true, y_pred, null_val=0.0):
    mask = _np_mask(y_true, null_val)
    mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true,
                            out=np.zeros_like(y_pred, dtype="float32"),
                            where=y_true != 0))
    return np.mean(np.nan_to_num(mape * mask)) * 100


def compute_metrics(y_true, y_pred, null_val=0.0):
    return (
        MAE_np(y_true, y_pred, null_val),
        RMSE_np(y_true, y_pred, null_val),
        MAPE_np(y_true, y_pred, null_val),
    )


def compute_all_metrics(y_true, y_pred, horizons=(3, 6, 12), null_val=0.0):
    res = {}
    mae, rmse, mape = compute_metrics(y_true, y_pred, null_val)
    res["avg"] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    res["horizon"] = {}
    for h in horizons:
        mae, rmse, mape = compute_metrics(y_true[:, h - 1], y_pred[:, h - 1], null_val)
        res["horizon"][h] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    res["per_step"] = []
    for i in range(y_true.shape[1]):
        mae, rmse, mape = compute_metrics(y_true[:, i], y_pred[:, i], null_val)
        res["per_step"].append({"MAE": mae, "RMSE": rmse, "MAPE": mape})
    return res


def _torch_mask(labels, null_val):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    return torch.nan_to_num(mask)


def masked_mae_loss(preds, labels, null_val=0.0):
    mask = _torch_mask(labels, null_val)
    loss = torch.abs(preds - labels) * mask
    return torch.mean(torch.nan_to_num(loss))


def masked_mse_loss(preds, labels, null_val=0.0):
    mask = _torch_mask(labels, null_val)
    loss = torch.square(preds - labels) * mask
    return torch.mean(torch.nan_to_num(loss))


def masked_huber_loss(preds, labels, null_val=0.0, delta=1.0):
    mask = _torch_mask(labels, null_val)
    err = torch.abs(preds - labels)
    quad = torch.clamp(err, max=delta)
    loss = (0.5 * quad ** 2 + delta * (err - quad)) * mask
    return torch.mean(torch.nan_to_num(loss))


LOSSES = {
    "masked_mae": masked_mae_loss,
    "masked_mse": masked_mse_loss,
    "masked_huber": masked_huber_loss,
}
