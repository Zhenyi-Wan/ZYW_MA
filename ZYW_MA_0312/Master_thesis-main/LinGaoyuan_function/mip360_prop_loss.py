import numpy as np

def inner_outer(t0, t1, y1):
  """Construct inner and outer measures on (t1, y1) for t0."""
  'jnp.zeros_like(y1[..., :1])只是多加0为第一个element, 然后正常y1array执行连加指令'
  cy1 = np.concatenate([np.zeros_like(y1[..., :1]),
                         np.cumsum(y1, axis=-1)],
                        axis=-1)
  '(idx_lo, idx_hi), where t1[idx_lo] <= t0 < t1[idx_hi]'
  idx_lo, idx_hi = searchsorted(t1, t0)

  'jnp.take_along_axis(): 根据idx选择cy1中的element, 也就是说这里是选出 propolsal mlp的权重array中满足idx_lo, idx_hi的元素'
  cy1_lo = np.take_along_axis(cy1, idx_lo, axis=-1)
  cy1_hi = np.take_along_axis(cy1, idx_hi, axis=-1)

  y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
  y0_inner = np.where(idx_hi[..., :-1] <= idx_lo[..., 1:],
                       cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
  return y0_inner, y0_outer


'论文公式13的propolsal loss'
'猜测: t应该是nerf mlp的ray distance, w是nerf mlp的weight, t_env和w_env应该是对应propolsal mlp'
def lossfun_outer(t, w, t_env, w_env, eps=np.finfo(np.float32).eps):
  """The proposal weight should be an upper envelope on the nerf weight."""
  _, w_outer = inner_outer(t, t_env, w_env)
  # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
  # more effective to pull w_outer up than it is to push w_inner down.
  # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
  return np.maximum(0, w - w_outer)**2 / (w + eps)


'这个函数的作用是计算v的element在插入到a的element中时应该在第几个index, 对应论文公式12时输入a是nerf mlp的t, v是propolsal mlp的t_env'


def searchsorted(a, v):
    """Find indices where v should be inserted into a to maintain order.

    This behaves like jnp.searchsorted (its second output is the same as
    jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
    faster because it wastes memory to save some compute.

    Args:
      a: tensor, the sorted reference points that we are scanning to see where v
        should lie.
      v: tensor, the query points that we are pretending to insert into a. Does
        not need to be sorted. All but the last dimensions should match or expand
        to those of a, the last dimension can differ.

    Returns:
      (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
      range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
      last index of a.
    """
    i = np.arange(a.shape[-1])
    v_ge_a = v[..., None, :] >= a[..., :, None]
    '''
    i[..., :, None]指对最底层的element多加一个维度,相当于对i的每个element单独加一个维度, 
    假设 i = [0,1,2,3], 则 i[..., :, None] = [[0],[1],[2],[3]], i[..., None, :] = [[0,1,2,3]]
    i[..., :1, None] = [[0]]
  
    jnp.where(condition,x,y), 指对x这个array中的每个element进行条件判断, 满足条件的保留, 不满足条件的element则用y这个variable替换
    假设condition是 i>2, 则jnp.where(i>2, i, 10) = [10,10,10,3]
  
    所以idx_lo是返回v中element比a中element大的index, 不满足部分用0取代, idx_hi是返回v中element比a中element小的index, 不满足部分用0取代
    下面jnp.max(...,-2)和jnp.min(...,-2)中的-2代表的是axis这个变量
    '''
    idx_lo = np.max(np.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)
    idx_hi = np.min(np.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)
    return idx_lo, idx_hi
