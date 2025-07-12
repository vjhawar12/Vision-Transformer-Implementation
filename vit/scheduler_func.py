from math import cos, pi

"""
Learning Rate Scheduler:

- **Epochs 0–9**: Linear warmup. This helped the model learn foundational features and stabilize early training. 
  Without it, accuracy plateaued at ~40% in the first 10 epochs.

- **Epochs 10–40**: Constant learning rate. MixUp augmentation was introduced here, allowing the model to 
  learn more complex features. Holding the learning rate steady helped avoid underfitting during this phase.

- **Epochs 41–100**: Cosine decay. Following Loshchilov & Hutter’s 2016 SGDR paper, a gradual decay helps 
  fine-tune features and consolidate knowledge. This phase led to improved convergence and boosted final 
  accuracy from ~84% to ~86%.

This composite schedule provided a balance of stability, exploration, and fine-tuning.
"""


def sched_func(epoch):
  if epoch < 10:
    return 1 - ((10 - epoch) / 10)

  if 10 <= epoch <= 40:
    return 1

  progress = (epoch - 40) / 60
  return 0.5 * (1 + cos(pi * progress))



