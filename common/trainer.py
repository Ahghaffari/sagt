import os
import time
import json
import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .metrics import LOSSES, compute_all_metrics
from .utils import JSONEncoder, count_parameters


class WarmupLR:
    def __init__(self, optimizer, d_model, n_warmup, lr_mul=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup = n_warmup
        self.lr_mul = lr_mul
        self.n_steps = 0

    def step(self):
        self.n_steps += 1
        lr = self.lr_mul * (self.d_model ** -0.5) * min(
            self.n_steps ** -0.5, self.n_steps * self.n_warmup ** -1.5)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


class CyclicCosineWarmupLR:
    def __init__(self, optimizer, d_model, n_warmup, lr_mul=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup = n_warmup
        self.lr_mul = lr_mul
        self.n_steps = 0

    def step(self):
        import math
        self.n_steps += 1
        if self.n_steps <= self.n_warmup:
            scale = (self.d_model ** -0.5) * min(
                self.n_steps ** -0.5, self.n_steps * self.n_warmup ** -1.5)
        else:
            scale = (self.d_model ** -0.5) * (self.n_warmup ** -0.5) * (
                1 + math.cos(math.pi * ((self.n_steps - self.n_warmup) % self.n_warmup)
                             / self.n_warmup))
        lr = self.lr_mul * scale
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


class Trainer:
    def __init__(self, model, loaders, scaler, cfg, run_dir, logger, device):
        self.model = model.to(device)
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.scaler = scaler
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = logger
        self.device = device

        tc = cfg["train"]
        self.criterion = LOSSES[tc.get("loss", "masked_mae")]
        if tc.get("loss") == "masked_huber" and "huber_delta" in tc:
            base, hd = self.criterion, tc["huber_delta"]
            self.criterion = lambda p, y, nv=0.0: base(p, y, nv, delta=hd)
        self.null_val = tc.get("null_val", 0.0)
        self.clip_grad = tc.get("clip_grad", None)
        self.max_epochs = tc.get("max_epochs", 200)
        self.patience = tc.get("early_stop", 30)

        opt = tc.get("optimizer", "adam").lower()
        if opt == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
        elif opt == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=tc["lr"],
                weight_decay=tc.get("weight_decay", 0.0), eps=tc.get("eps", 1e-8))
        else:
            # betas configurable: Transformer/Noam-scheduled models need (0.9, 0.98)
            # -- with a cyclic/warmup LR, beta2=0.999 adapts far too slowly.
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=tc["lr"],
                betas=tuple(tc.get("betas", (0.9, 0.999))),
                weight_decay=tc.get("weight_decay", 0.0), eps=tc.get("eps", 1e-8))
        self.per_batch_sched = False
        self.scheduler = self._build_scheduler(tc.get("scheduler"))
        self.writer = SummaryWriter(run_dir)
        self.ckpt_path = os.path.join(run_dir, "best.pt")

        self.out_steps = cfg["data"]["out_steps"]
        self.batches_seen = 0
        self.curriculum = tc.get("curriculum")
        self.task_level = 0 if self.curriculum else None
        self.loss_is_aux = getattr(self.model, "loss_is_aux", False)
        if hasattr(self.model, "set_scaler"):
            self.model.set_scaler(scaler)

    def _build_scheduler(self, sc):
        if not sc:
            return None
        t = sc["type"]
        if t == "warmup":
            self.per_batch_sched = True
            return WarmupLR(self.optimizer, d_model=sc["d_model"],
                            n_warmup=sc["n_warmup_steps"], lr_mul=sc.get("lr_mul", 1.0))
        if t == "cosine_warmup_cyclic":
            self.per_batch_sched = True
            return CyclicCosineWarmupLR(self.optimizer, d_model=sc["d_model"],
                                        n_warmup=sc["n_warmup_steps"],
                                        lr_mul=sc.get("lr_mul", 1.0))
        if t == "multistep":
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=sc["milestones"], gamma=sc.get("gamma", 0.1))
        if t == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sc["step_size"], gamma=sc.get("gamma", 0.1))
        if t == "cosine":
            w = sc.get("warmup_epochs", 0)
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sc.get("t_max", self.max_epochs) - w,
                eta_min=sc.get("eta_min", 0.0))
            if w > 0:
                lin = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=sc.get("warmup_init_factor", 1e-3),
                    total_iters=w)
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, [lin, cos], milestones=[w])
            return cos
        if t == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sc.get("gamma", 0.97))
        raise ValueError(f"unknown scheduler {t}")

    def _forward(self, x, y, training):
        label = self.scaler.transform(y)
        if training and self.curriculum is not None:
            if (self.batches_seen % self.curriculum["step_size"] == 0
                    and self.task_level < self.out_steps):
                self.task_level += 1
        if self.curriculum is not None:
            tl = self.task_level if training else self.out_steps
        else:
            tl = None
        out = self.model(x, label=label, batches_seen=self.batches_seen, task_level=tl)
        aux = 0.0
        if isinstance(out, tuple):
            out, aux = out
        out = self.scaler.inverse_transform(out)
        if training and self.curriculum is not None:
            out, y = out[:, :self.task_level], y[:, :self.task_level]
        return out, y, aux

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out, y_t, aux = self._forward(x, y, training=True)
            if self.loss_is_aux:
                loss = aux
            else:
                loss = self.criterion(out, y_t, self.null_val) + aux
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            if self.per_batch_sched:
                self.scheduler.step()
            self.optimizer.step()
            self.batches_seen += 1
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        losses = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out, y_t, aux = self._forward(x, y, training=False)
            loss = self.criterion(out, y_t, self.null_val) + aux
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        ys, outs = [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out, y_t, _ = self._forward(x, y, training=False)
            outs.append(out.cpu().numpy())
            ys.append(y_t.cpu().numpy())
        y = np.vstack(ys).squeeze(-1)
        out = np.vstack(outs).squeeze(-1)
        return y, out

    def train(self):
        self.logger.info(f"params: {count_parameters(self.model):,}")
        best_val = np.inf
        best_state = None
        best_epoch = 0
        wait = 0
        history = []
        t0 = time.time()
        for epoch in range(1, self.max_epochs + 1):
            te = time.time()
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)
            train_loss = self.train_epoch()
            val_loss = self.eval_epoch(self.val_loader)
            if self.scheduler and not self.per_batch_sched:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - te
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("lr", lr, epoch)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "lr": lr, "time": dt})
            self.logger.info(
                f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {lr:.2e} | {dt:.1f}s")
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.logger.info(f"early stop at epoch {epoch}")
                    break
        total_time = time.time() - t0
        self.model.load_state_dict(best_state)
        torch.save(best_state, self.ckpt_path)
        self.logger.info(f"best epoch {best_epoch} val {best_val:.4f} | "
                         f"train time {total_time:.0f}s")
        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, cls=JSONEncoder)
        return {"best_epoch": best_epoch, "best_val_loss": best_val,
                "train_time_s": total_time,
                "epochs_run": len(history),
                "params": count_parameters(self.model)}

    def test(self, train_summary=None):
        horizons = tuple(self.cfg["train"].get("report_horizons", [3, 6, 12]))
        t0 = time.time()
        y, out = self.predict(self.test_loader)
        infer_time = time.time() - t0
        res = compute_all_metrics(y, out, horizons=horizons, null_val=self.null_val)
        a = res["avg"]
        self.logger.info(
            f"TEST avg | MAE {a['MAE']:.4f} RMSE {a['RMSE']:.4f} MAPE {a['MAPE']:.4f}%")
        for h in horizons:
            m = res["horizon"][h]
            self.logger.info(
                f"  h{h:2d} | MAE {m['MAE']:.4f} RMSE {m['RMSE']:.4f} MAPE {m['MAPE']:.4f}%")
        metrics = {"test": res, "infer_time_s": infer_time}
        if train_summary:
            metrics["train"] = train_summary
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, cls=JSONEncoder)
        return metrics
