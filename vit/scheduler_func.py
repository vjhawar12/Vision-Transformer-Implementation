from math import cos, pi

def sched_func(epoch):
  if epoch < 10:
    return 1 - ((10 - epoch) / 10)

  if 10 <= epoch <= 40:
    return 1

  progress = (epoch - 40) / 60
  return 0.5 * (1 + cos(pi * progress))



