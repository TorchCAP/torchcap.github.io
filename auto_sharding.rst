Auto Sharding
=============

This document aims to describe the automatic sharding of TorchCAP. It provides the implementation details of automatic sharding, including an overview of the automatic sharding optimizer, the optimization formulation of finding an optimal sharding plan and the sharding transformation of the model graph and.

Automatic Sharding Overview
^^^^^^^^^^^^^^^^^^^^^^

The high-level API of automatic sharding is ``torchcap.optimize`` (`link <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/api.py>`_), which performs the following steps:

1. Convert an input model to an FX graph via ``torch.export``
2. Estimate the runtime and memory consumption of each operator in the graph
3. Find the optimal sharding strategy for the graph
4. Transform the graph into a distributed graph using the sharding strategy

Automatic Sharding Solver
^^^^^^^^^^^^^^^^^^^^^^

The formulation used in the solver is based on the integer linear programming (ILP) formulation proposed in `Alpa <https://arxiv.org/abs/2201.12023>`_, with modifications for Pytorch DTensor support. The implementation can be found in `parallel_solver.py <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/solver/parallel_solver.py>`_.

Pytorch DTensor supports three types of ``sharding`` (also refer as ``placement`` in Pytorch official documentation): ``Shard(dim)`` (``R``) means the sharding of tensor dimension ``dim`` over the mesh dimension; ``Replicate()`` (``R``) means the tensor is replicated over the mesh dimension; ``Partial`` (``P``) indicates the tensor is pending reduction on the devices of the. For a N-dimensional mesh, a N-dimensional vector is used to represent the sharding over each mesh dimension. For example, `S(0)R` indicates the dimension 0 is sharded over the mesh dimension 0 and is replicated over the mesh dimension 1. More details can be found in `Pytorch official documentation <https://pytorch.org/docs/stable/distributed.tensor.html>`_.

For each operator in the graph, the solver enumerates all possible sharding strategies based on the operator sharding rules defined in `Pytorch <https://github.com/pytorch/pytorch/tree/4273e5d15cfcb282b2795684874ea439d8620999/torch/distributed/tensor/_ops>`_ and `sharding_strategy.py <https://github.com/TorchCAP/TorchCAP/blob/6abd50d1a31b0bdf4762c914cf5e583d3810117d/torchcap/solver/sharding_strategy.py>`_. For example, a sharding strategy of a linear operator ``R, S(0), S(0) -> S(1)``, representing that first argument is replicated, the second and third arguments are sharded over the tensor dimension 0 and the output is sharded over the tensor dimension 1. For an operator ``u``, a one-hot vector ``s_u`` is used to represent the strategy it uses, where ``s_u[i] = 1`` means the i-th strategy of operator ``u`` is selected.

When two operators require different shardings for the same tensor, additional communication may be necessary for resharding. For example, the output of the linear operator is ``S(1)`` but the operator used this output requires it to be ``R``, then a all-gather communication is needed to reshard the tensor, which is the cause of communication overheads in sharding. For the resharding cost between operator ``u`` and operator ``v``, the solver constructs a resharding cost
matrix :math:`R_{uv}`, where :math:`R_{uv}[i][j]` is the resharding cost from the output of i-th strategy of node u to the input of j-th strategy of node v.

The objective of the formulation is 

.. math::

  \sum_{(u,v) \in E} s_u R_{uv}[i][j] s_v

This formulation is similar to the one proposed in Alpa but without communication cost as it is already been included in the resharding cost.

Additionally, the formulation has a memory constraint to limit the memory usage of each device, which is defined as the maximum memory consumption of each GPU.

Let :math:`u_0,u_1,\cdots,u_{n-1}` be the set of operators in the graph represented in topological order, and :math:`m_t` be the memory consumption of output tensor of operator :math:`u_t`. Then :math:`m_t` can be computed by

.. math::

  m_t = \sum_{i} s_i \cdot \text{output_size}(u_t)[i]

where :math:`\text{output_size}(u_t)[i]` is the output tensor size of operator :math:`u_i` with the k-th strategy.

By analyzing the graph, we can extract the live range of each output tensor, represented by :math:`[start_k, end_k]`. Let :math:`delta[t]` denotes the memory change during the execution of operator :math:`t`.

.. math::

  delta[t] = m_t - \sum_{\forall k, t = end_k} m_i

where the first term is the memory allocation of the output tensor of operator :math:`t` and the second term is the memory deallocation of the output tensor last used by operator :math:`t`.


Then the accumulated memory consumption :math:`M_t` during each operator :math:`t` can be computed by

.. math::

  M_t = M_{t-1} + delta[t]


And the memory constraint can be formulated as

.. math::

  \max_{t} M_t \leq \text{max_memory}


Sharding Transformation
^^^^^^^^^^^^^^^^^^^^^^

