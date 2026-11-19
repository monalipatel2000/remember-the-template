"""EMC, the exemplar-memory classifier, as a small reusable module. Same computation and settings as final_eval.py.

    model = EMC(seed=13).fit(train_image, train_text, train_labels)     # L2-normalised CLIP embeddings, (n, 768) each
    probs = model.predict_proba(query_image, query_text)
    probs, neighbours = model.predict_proba(query_image, query_text, return_neighbours=True)   # indices of the voters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMC:
    def __init__(self, k=16, tau=.05, head_weight=.5, hidden=384, epochs=40, batch=256, lr=5e-4, seed=13, device=None):
        self.k, self.tau, self.lam, self.hidden, self.epochs, self.batch, self.lr, self.seed = k, tau, head_weight, hidden, epochs, batch, lr, seed
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _t(self, a, dtype=None):
        return torch.as_tensor(a.copy() if hasattr(a, 'copy') else a, device=self.dev, dtype=dtype)

    def fit(self, image, text, labels, n_classes=None):
        """Train the head on [image, text]; store the image embeddings and labels as the memory. The memory has no parameters."""
        self.keys, self.y = self._t(image, torch.float32), self._t(labels, torch.long)
        X = torch.cat([self.keys, self._t(text, torch.float32)], 1); y = self.y
        self.n_classes = n_classes or int(y.max()) + 1
        torch.manual_seed(self.seed)
        self.head = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(.3), nn.Linear(self.hidden, self.n_classes)).to(self.dev)
        opt = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epochs * ((len(y) + self.batch - 1) // self.batch))
        for _ in range(self.epochs):
            self.head.train(); perm = torch.randperm(len(y), device=self.dev)
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                loss = F.cross_entropy(self.head(X[b]), y[b], label_smoothing=.1)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        self.head.eval()
        return self

    def add_exemplars(self, image, labels):
        """Extend the memory with new labelled memes. No retraining."""
        self.keys = torch.cat([self.keys, self._t(image, torch.float32)]); self.y = torch.cat([self.y, self._t(labels, torch.long)])
        return self

    @torch.no_grad()
    def memory_proba(self, image):
        sim, idx = (self._t(image, torch.float32) @ self.keys.T).topk(min(self.k, len(self.y)), dim=1)
        p = torch.zeros(len(idx), self.n_classes, device=self.dev).scatter_add_(1, self.y[idx], F.softmax(sim / self.tau, 1))
        return p, idx

    @torch.no_grad()
    def predict_proba(self, image, text, return_neighbours=False):
        h = F.softmax(self.head(torch.cat([self._t(image, torch.float32), self._t(text, torch.float32)], 1)), 1)
        m, idx = self.memory_proba(image)
        p = self.lam * h + (1 - self.lam) * m
        return (p, idx) if return_neighbours else p
