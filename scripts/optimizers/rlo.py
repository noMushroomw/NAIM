import math

import torch
from torch.optim import Optimizer

def _finite(t):
    return not (torch.isnan(t).any() or torch.isinf(t).any())

class RLO(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, lambda_b=0.2):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, lambda_b=lambda_b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None or not _finite(p.grad):
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                c = exp_avg * beta1 + grad * (1 - beta1)
                delta = grad - exp_avg
                update = torch.sign(c) + group['lambda_b'] * (delta / delta.norm().clamp(min=1e-8))
                p.add_(update, alpha=-group['lr'])
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss

class RLOLambda(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), beta3=0.99, weight_decay=0.0,
                 lambda_b=0.2, gamma=5.0, eps=1e-8, scale_mode='grad_ema', scale_cap=10.0):
        if scale_mode not in ('grad_ema', 'sqrt_dim'):
            raise ValueError(f"scale_mode must be 'grad_ema' or 'sqrt_dim', got {scale_mode!r}")
        defaults = dict(lr=lr, betas=betas, beta3=beta3, weight_decay=weight_decay,
                        lambda_b=lambda_b, gamma=gamma, eps=eps)
        super().__init__(params, defaults)
        self.scale_mode = scale_mode
        self.scale_cap = float(scale_cap)
        self.global_state = {'smooth_norm': 1.0, 'scale': None}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        valid = [(group, p) for group in self.param_groups for p in group['params']
                 if p.grad is not None and _finite(p.grad)]
        if not valid:
            return loss
        total_dim = sum(p.numel() for _, p in valid)

        updates = []
        for group, p in valid:
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            beta1, _ = group['betas']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg_sq.mul_(group['beta3']).addcmul_(grad, grad, value=1 - group['beta3'])
            c = exp_avg * beta1 + grad * (1 - beta1)
            direction = torch.tanh(group['gamma'] * c / exp_avg_sq.sqrt().add_(group['eps']))
            delta = grad - exp_avg
            belief = group['lambda_b'] * (delta / delta.norm().clamp(min=1e-8))
            updates.append(direction + belief)

        if self.scale_mode == 'grad_ema':
            grad_norm = math.sqrt(sum(p.grad.norm().pow(2).item() for _, p in valid) + 1e-8)
            self.global_state['smooth_norm'] = 0.99 * self.global_state['smooth_norm'] + 0.01 * grad_norm
            scale = min(math.sqrt(total_dim) / max(self.global_state['smooth_norm'], 1e-8), self.scale_cap)
        else:
            u_norm = math.sqrt(torch.stack([u.pow(2).sum() for u in updates]).sum().item())
            scale = math.sqrt(total_dim) / (u_norm + 1e-8)
        self.global_state['scale'] = scale

        for (group, p), u in zip(valid, updates):
            p.add_(scale * u, alpha=-group['lr'])
            if group['weight_decay'] > 0:
                p.add_(p, alpha=-group['lr'] * group['weight_decay'])
            _, beta2 = group['betas']
            self.state[p]['exp_avg'].mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss

class RLOLifted(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eta=0.7, weight_decay=0.0,
                 lambda_b=0.2, gamma=5.0, eps=1e-8, field='tanh', use_global_norm=True,
                 belief_norm='tensor', collect_diagnostics=False, naim_interval=0):
        if field not in ('tanh', 'sign'):
            raise ValueError(f"field must be 'tanh' or 'sign', got {field!r}")
        if belief_norm not in ('tensor', 'global'):
            raise ValueError(f"belief_norm must be 'tensor' or 'global', got {belief_norm!r}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f'eta must lie in (0, 1], got {eta}')
        defaults = dict(lr=lr, betas=betas, eta=eta, weight_decay=weight_decay,
                        lambda_b=lambda_b, gamma=gamma, eps=eps)
        super().__init__(params, defaults)
        self.field = field
        self.use_global_norm = use_global_norm
        self.belief_norm = belief_norm
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g['params']))
        self.collect_diagnostics = collect_diagnostics
        self.diagnostics = {}
        self.naim_interval = int(naim_interval)
        self.naim_log = {}
        self._naim_has_prev = False
        self._k = 0

    def get_diagnostics(self):

        diag, self.diagnostics = self.diagnostics, {}
        return diag

    def _log(self, key, value):
        self.diagnostics.setdefault(key, []).append(value)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        entries = []
        for group in self.param_groups:
            beta1, _ = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                g, m = p.grad, state['m']
                c = beta1 * m + (1 - beta1) * g
                s = c.sign() if self.field == 'sign' else torch.tanh(group['gamma'] * c)
                entries.append((p, group, state, s, g - m))
        if not entries:
            return loss

        if self.belief_norm == 'global':
            global_delta = math.sqrt(sum(float((d * d).sum()) for *_, d in entries))
            denoms = [global_delta + entries[0][1]['eps']] * len(entries)
        else:
            denoms = [d.norm().clamp(min=group['eps']) for _, group, _, _, d in entries]

        if self.use_global_norm:
            s_norm = math.sqrt(sum((s * s).sum().item() for _, _, _, s, _ in entries) + 1e-8)
            scale = self.sqrt_dim / s_norm
        else:
            scale = 1.0

        collect = self.collect_diagnostics
        naim = self.naim_interval > 0 and self._k % self.naim_interval == 0
        has_prev = naim and self._naim_has_prev
        sq = dict(res=0.0, vd=0.0, vv=0.0, dd=0.0, bb=0.0)
        nq = dict(zz=0.0, vd=0.0, vv=0.0, dd=0.0, qq=0.0, qd=0.0)
        for (p, group, state, s, delta), denom in zip(entries, denoms):
            if group['weight_decay'] != 0:
                p.mul_(1 - group['lr'] * group['weight_decay'])
            b = group['lambda_b'] * (delta / denom)
            d = scale * s + b
            v = state['v']
            if naim:

                z = v - d
                nq['zz'] += float((z * z).sum())
                nq['vd'] += float((v * d).sum())
                nq['vv'] += float((v * v).sum())
                nq['dd'] += float((d * d).sum())
                if has_prev:
                    q = d - state['naim_prev_d']
                    nq['qq'] += float((q * q).sum())
                    nq['qd'] += float((q * d).sum())
                    state['naim_prev_d'].copy_(d)
                else:
                    state['naim_prev_d'] = d.clone()
            v.mul_(1 - group['eta']).add_(d, alpha=group['eta'])
            p.add_(v, alpha=-group['lr'])
            _, beta2 = group['betas']
            state['m'].mul_(beta2).add_(p.grad, alpha=1 - beta2)
            if collect:
                sq['res'] += float(((v - d) ** 2).sum())
                sq['vd'] += float((v * d).sum())
                sq['vv'] += float((v * v).sum())
                sq['dd'] += float((d * d).sum())
                sq['bb'] += float((b * b).sum())

        if collect:
            self._log('scale', scale)
            self._log('fiber_residual', math.sqrt(sq['res']))
            self._log('fiber_alignment', sq['vd'] / max(math.sqrt(sq['vv'] * sq['dd']), 1e-12))
            self._log('d_norm', math.sqrt(sq['dd']))
            self._log('v_norm', math.sqrt(sq['vv']))
            self._log('belief_norm', math.sqrt(sq['bb']))
        if naim:
            self._log_naim(nq, has_prev)
        self._k += 1
        return loss

    def _log_naim(self, nq, has_prev, eps=1e-8):
        d_norm, v_norm, z_norm = math.sqrt(nq['dd']), math.sqrt(nq['vv']), math.sqrt(nq['zz'])
        rec = {'step': self._k, 'd_norm': d_norm, 'v_norm': v_norm, 'z_norm': z_norm,
               'r_rel': z_norm / (d_norm + eps),
               'cos_vd': nq['vd'] / (v_norm * d_norm + eps),
               'drift_norm': None, 'q_perp_rel': None}
        if has_prev:
            perp_sq = max(nq['qq'] - nq['qd'] ** 2 / (nq['dd'] + eps), 0.0)
            rec['drift_norm'] = math.sqrt(nq['qq'])
            rec['q_perp_rel'] = math.sqrt(perp_sq) / (d_norm + eps)
        for key, val in rec.items():
            self.naim_log.setdefault(key, []).append(val)
        self._naim_has_prev = True

    def pop_naim_log(self):

        log, self.naim_log = self.naim_log, {}
        return log
