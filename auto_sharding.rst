Auto Sharding
=============

This document describes the automatic sharding mechanism in TorchCAP, detailing the implementation of the auto-sharding pipeline. It covers the architecture of the sharding optimizer, the formulation used to determine an optimal sharding plan, and the transformation process applied to the model graph.

Automatic Sharding Overview
^^^^^^^^^^^^^^^^^^^^^^

The high-level API for automatic sharding is available via ``torchcap.optimize`` (`api.py <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/api.py>`_). The optimizer performs the following steps:

1. Converts the input model into an FX graph using torch.export

2. Estimates the runtime and memory consumption of each operator in the graph

3. Determines the optimal sharding strategy for the graph

4. Transforms the model into a distributed version using the selected strategy

Users may also provide custom sharding strategies for specific operations. The optimizer will compute the optimal strategy for the remaining parts of the graph accordingly.


Automatic Sharding Solver
^^^^^^^^^^^^^^^^^^^^^^

The solver formulation is based on the integer linear programming (ILP) approach proposed in `Alpa <https://arxiv.org/abs/2201.12023>`_, with modifications to support PyTorch DTensor. The implementation is available in `parallel_solver.py <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/solver/parallel_solver.py>`_.

PyTorch DTensor supports three types of ``sharding`` (also referred to as ``placement`` in the PyTorch documentation):

- ``Shard(dim)`` (``S(dim)``): Shards the specified tensor dimension over the mesh dimension.
- ``Replicate`` (``R``): Replicates the tensor across the mesh dimension.
- ``Partial`` (``P``): Indicates the tensor is pending reduction across devices.

For an N-dimensional mesh, a vector of length N represents the sharding strategy across each dimension. For example, ``S(0)R`` indicates dimension 0 is sharded over mesh dimension 0 and replicated over mesh dimension 1. See the `PyTorch DTensor documentation <https://pytorch.org/docs/stable/distributed.tensor.html>`_ for further details.

For each operator in the graph, the solver enumerates all possible sharding strategies based on the operator sharding rules defined in both `Pytorch <https://github.com/pytorch/pytorch/tree/4273e5d15cfcb282b2795684874ea439d8620999/torch/distributed/tensor/_ops>`_ and `sharding_strategy.py <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/solver/sharding_strategy.py>`_. For example, a sharding strategy of a linear operator ``R, S(0), S(0) -> S(1)``, representing that first argument is replicated, the second and third arguments are sharded over the tensor dimension 0 and the output is sharded over the tensor dimension 1. 

Each operator ``u`` is assigned a one-hot vector :math:`s_u`, where :math:`s_u[i] = 1` denotes that the i-th strategy has been selected for operator ``u``.

When two operators require incompatible sharding for a shared tensor, communication is needed for resharding. For instance, if the output of a linear operator is ``S(1)``, but the consumer operator expects it as ``R``, an all-gather operation is required—introducing communication overhead. For the resharding cost between operator ``u`` and operator ``v``, the solver constructs a resharding cost matrix :math:`R_{uv}`, where :math:`R_{uv}[i][j]` is the cost of resharding the output of strategy i for operator ``u`` into the input of strategy j for operator ``v``.

The objective of the formulation is 

.. math::

  \sum_{(u,v) \in E} s_u R_{uv}[i][j] s_v

This formulation captures both the choice of strategy and the communication cost, similar to Alpa’s formulation but with communication overheads folded into the resharding costs.

Memory Constraint
+++++++++++++

To bound memory usage per device, a memory constraint is added to the formulation. Let :math:`u_0, u_1, \ldots, u_{n-1}` be the operators in topological order. Let :math:`m_t` be the memory consumed by the output tensor of operator :math:`u_t`. It is calculated as:

.. math::

  m_t = \sum_{i} s_i \cdot \text{output_size}(u_t)[i]

Here, :math:`\text{output\_size}(u_t)[i]` is the output size of operator :math:`u_t` under strategy i.

Using liveness analysis, we extract the live range of each output tensor as :math:`[start_k, end_k]`. Define :math:`\delta[t]` as the net memory change at step :math:`t`:

.. math::

  delta[t] = m_t - \sum_{\forall k, t = end_k} m_i

where the first term is the memory allocation of the output tensor of operator :math:`t` and the second term is the memory deallocation of the output tensor last used by operator :math:`t`.

The cumulative memory consumption at step :math:`t`, denoted as :math:`M_t`, is then:

.. math::

  M_t = M_{t-1} + delta[t]


Therefore, the memory constraint can be represented as

.. math::

  \max_{t} M_t \leq \text{max_memory}


Sharding Transformation
^^^^^^^^^^^^^^^^^^^^^^

The sharding transformation pass is implemented in `tensor_parallel.py <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/transform/tensor_parallel.py>`_.

The pass performs the following steps:

1. Annotate the sharding strategy for each operator and derive the sharding strategy (``_mark_tensor_parallel_shardings``)
2. Partition the single device graph to distributed graph (``_partitioner``)
  1. Insert the resharding communication operations if there is a misaligned sharding (``_insert_reshard_gm``)
3. Partition the parameters based on the sharding strategy  (``_shard_state_dict``)
