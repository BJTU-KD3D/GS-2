import logging
from collections import deque

import torch


class ELBOController:


    def __init__(
        self,
        elbo_threshold=1e-4,
        lambda_xi=0.01,
        window_size=5,
        min_iter=6000,
        patience=5,
        ema_alpha=0.95,
        log_file=None,
    ):
        self.elbo_threshold = float(elbo_threshold)
        self.lambda_xi = float(lambda_xi)
        self.window_size = int(window_size)
        self.min_iter = int(min_iter)
        self.patience = int(patience)
        self.ema_alpha = float(ema_alpha)

        self.elbo_history = deque(maxlen=max(800, self.window_size * self.patience + 1))
        self.best_elbo = -float("inf")
        self.no_improvement_count = 0
        self.stable_windows = 0
        self.adaptive_stop_iter = None
        self.prev_elbo_ema = None
        self.densify_disabled = False

        self.logger = None
        if log_file:
            logger = logging.getLogger("ELBO")
            logger.setLevel(logging.INFO)
            logger.handlers.clear()

            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
            self.logger = logger

    def _safe_device(self, gaussians):
        if hasattr(gaussians, "_xyz") and gaussians._xyz is not None:
            return gaussians._xyz.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_normalized_density(self, gaussians):

        if not hasattr(gaussians, "_xyz") or gaussians._xyz is None or gaussians._xyz.shape[0] == 0:
            device = self._safe_device(gaussians)
            return torch.tensor(0.0, device=device)

        xyz = gaussians._xyz  # expected shape: [N, 3]
        num_points = xyz.shape[0]

        xyz_min = xyz.min(dim=0).values
        xyz_max = xyz.max(dim=0).values
        bbox_size = (xyz_max - xyz_min).clamp(min=1e-8)
        bbox_volume = bbox_size.prod()

        xi_raw = num_points / (bbox_volume + 1e-8)

        # Normalize density for stability.
        # This keeps the scale from exploding across scenes.
        xi_norm = xi_raw / (num_points + 1e-8)

        return xi_norm

    def compute_normalized_covariance_term(self, gaussians):

        if not hasattr(gaussians, "_scaling") or gaussians._scaling is None or gaussians._scaling.shape[0] == 0:
            device = self._safe_device(gaussians)
            return torch.tensor(0.0, device=device)

        scaling = gaussians._scaling  # expected shape: [N, 3] or compatible
        sigma_diag = torch.exp(scaling) + 1e-8

        if sigma_diag.ndim != 2 or sigma_diag.shape[1] != 3:
            raise ValueError(
                f"Expected gaussians._scaling to have shape [N, 3], but got {tuple(sigma_diag.shape)}"
            )

        # Normalize covariance by batch-wise mean
        sigma_mean = sigma_diag.mean(dim=0, keepdim=True) + 1e-8
        sigma_tilde = sigma_diag / sigma_mean

        trace_term = sigma_tilde.sum(dim=1)          # [N]
        logdet_term = torch.log(sigma_tilde).sum(dim=1)  # [N], diagonal covariance

        cov_term = 0.5 * (trace_term - logdet_term)
        cov_term = cov_term.mean()

        return cov_term

    def compute_model_complexity(self, gaussians):

        device = self._safe_device(gaussians)

        cov_term = self.compute_normalized_covariance_term(gaussians)
        xi_norm = self.compute_normalized_density(gaussians)
        density_term = self.lambda_xi * torch.log1p(xi_norm)

        total = cov_term + density_term
        if not torch.isfinite(total):
            return torch.tensor(0.0, device=device)

        return total

    def compute_elbo(self, loss, gaussians):

        if isinstance(loss, torch.Tensor):
            loss_val = loss.item()
        else:
            loss_val = float(loss)

        complexity = self.compute_model_complexity(gaussians).item()
        elbo_value = -loss_val - complexity
        return float(elbo_value)

    def update_and_check_convergence(self, elbo, iteration, gaussians=None, loss=None):

        elbo = float(elbo)

        if self.prev_elbo_ema is None:
            elbo_ema = elbo
        else:
            elbo_ema = self.ema_alpha * self.prev_elbo_ema + (1.0 - self.ema_alpha) * elbo

        self.prev_elbo_ema = elbo_ema
        self.elbo_history.append(elbo_ema)

        if elbo_ema > self.best_elbo:
            self.best_elbo = elbo_ema
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if iteration % 100 == 0 and self.logger is not None:
            msg = f"ITER {iteration}: elbo={elbo:.6f}, elbo_ema={elbo_ema:.6f}"
            if gaussians is not None:
                complexity = self.compute_model_complexity(gaussians).item()
                xi_norm = self.compute_normalized_density(gaussians).item()
                cov_term = self.compute_normalized_covariance_term(gaussians).item()
                msg += (
                    f", complexity={complexity:.6f}, "
                    f"cov_term={cov_term:.6f}, density={xi_norm:.6f}"
                )
            if loss is not None:
                if isinstance(loss, torch.Tensor):
                    msg += f", loss={loss.item():.6f}"
                else:
                    msg += f", loss={float(loss):.6f}"
            msg += f", no_improve={self.no_improvement_count}"
            self.logger.info(msg)

        if iteration < self.min_iter:
            return False

        if len(self.elbo_history) >= self.window_size and (iteration % self.window_size == 0):
            delta_t = abs(self.elbo_history[-1] - self.elbo_history[-self.window_size]) / (
                abs(self.elbo_history[-1]) + 1e-6
            )

            if delta_t < self.elbo_threshold:
                self.stable_windows += 1
            else:
                self.stable_windows = 0

            if self.stable_windows >= self.patience:
                if self.adaptive_stop_iter is None:
                    self.adaptive_stop_iter = iteration
                    print(f"[ITER {iteration}] ELBO converged (delta={delta_t:.6e}), stopping densification")
                self.densify_disabled = True
                return True

        # Optional plateau fallback
        if self.no_improvement_count >= self.patience * self.window_size:
            if self.adaptive_stop_iter is None:
                self.adaptive_stop_iter = iteration
                print(f"[ITER {iteration}] ELBO plateau, stopping densification")
            self.densify_disabled = True
            return True

        return False

    def get_adaptive_densify_until_iter(self, original_densify_until_iter):

        original_densify_until_iter = int(original_densify_until_iter)

        if self.densify_disabled and self.adaptive_stop_iter is not None:
            return min(self.adaptive_stop_iter, original_densify_until_iter)

        return original_densify_until_iter

    def get_adaptive_densify_grad_threshold(self, base_threshold, iteration=None):

        return float(base_threshold)

    def reset(self):
 
        self.elbo_history.clear()
        self.best_elbo = -float("inf")
        self.no_improvement_count = 0
        self.stable_windows = 0
        self.adaptive_stop_iter = None
        self.prev_elbo_ema = None
        self.densify_disabled = False
