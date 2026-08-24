自适应点对点通信
===================================

该组件在 PyTorch ``torch.distributed`` 之上为 Worker 之间提供 **严格顺序** 和 **异步句柄** 的点对点 (P2P) 数据传输。  
它包含两个对外的类：

- **Collective**：每个 Worker 的单例，用于创建/缓存通信组。  
- **CollectiveGroup**：一个两节点的通信组，实现了张量、张量列表/字典以及可序列化 Python 对象的 P2P send/recv。  


组的创建与缓存
----------------------------------------

``Collective`` 类在每个 Worker 上实例化（每个 Worker 一个单例），  
负责创建并缓存 ``CollectiveGroup`` 实例。  
当两个 Worker 或一组 Worker 需要通信时，必须建立一个包含所有参与者的 collective group。  
在本框架中的典型用法是通过  
``Collective.create_collective_group(worker_addresses, group_name=None)``  
来形成点对点通信组，该方法会返回给定 Worker 地址集合的现有 ``CollectiveGroup``，或者新建一个。  


.. _collectivegroup_p2p:

点对点通信
-------------------------------------

``CollectiveGroup`` 是 RLinf 中管理两个 Worker 间点对点通信的核心抽象。  
它会根据 ``group_info`` 确定本地 rank（0 或 1），并在首次使用时 **延迟初始化** 通信进程组。  

在内部，会分别为 GPU (NCCL) 和 CPU (Gloo) 创建独立的 **发送** 和 **接收** 进程组，形成专用的单向通道；  
在双 Worker 设置中，精心配置的广播等价于 send/recv。  
初始化过程使用 TCP rendezvous 协调端口分配与同步，确保双方准备就绪。  
每个方向都有一个基于专用 CUDA stream 的工作队列，严格保证 send/recv 操作的顺序，避免消息交错。  

建立进程组后，``CollectiveGroup`` 可以执行通信。主要 API 有：

- **Send**: ``send(obj, async_op=False)``  
  向组内的另一方发送一个对象（张量、张量列表、张量字典或任意可序列化对象）。  
  此方法会先发送一个小的 **header**，指明对象类型，以便接收端正确解析负载。  

- **Recv**: ``recv(async_op=False)``  
  从对端接收一个对象。它首先接收类型码（CPU/Gloo），然后调用相应的接收器重建对象。  

- **Direct Tensor Send/Recv**: ``send_tensor(tensor, async_op=False)`` 与 ``recv_tensor(tensor, async_op=False)``  
  针对仅传输单个张量且接收端已分配好张量缓冲区的情况进行了优化，避免了额外的元数据往返。  

.. note::
   所有 **CUDA 张量必须是连续的**；非连续张量会触发错误提示。  
   不允许在同一列表/字典中混合 CPU 与 CUDA 张量。  

.. warning::
   ``send_tensor`` **必须** 与 ``recv_tensor`` 配对使用（反之亦然）。  
   不要在同一消息中将它们与通用的 ``send``/``recv`` 混用。  


Tensor 压缩
---------------------------------

当网络传输时间高于额外的 CPU 开销时，可以启用无损 CPU tensor 压缩。压缩是可选的
作业级能力：driver 校验一份配置，并将其下发到所有 Worker。

适用范围
~~~~~~~~

压缩仅作用于通用 ``Worker.send``/``Worker.recv`` 优化路径携带的 CPU tensor：单个
tensor、tensor list 或 tuple、值全为 tensor 的 dictionary，以及从 dataclass 中提取的
tensor 字段。对于同时包含 CPU 和 accelerator tensor 的容器，只有 CPU tensor 会成为
候选。通过 Worker send/recv 通信的 Channel 会继承相同行为。

任意 pickled Python object、accelerator tensor、``broadcast``，以及直接调用
``send_tensor``/``recv_tensor`` 的路径不会压缩。tensor list 的元素仍然是独立的 wire
payload；该功能只减少每个 payload 的字节数，不会将它们拼凑起来。

选择 Codec
~~~~~~~~~~

两种 codec 都是无损的。它们处理连续 CPU tensor 的原始字节，并逐字节恢复原始 dtype 和
shape。压缩和解压缩直接在 tensor 与预分配的 ``torch.uint8`` buffer 之间进行，不会创建
中间 Python ``bytes`` object。

.. list-table:: Codec 取舍
   :header-rows: 1
   :widths: 14 30 24 32

   * - Codec
     - 特征
     - ``level``
     - 适用情况
   * - ``lz4``
     - 优先保证压缩和解压缩速度，CPU 开销相对较低，但压缩率通常低于 Zstd。
     - 作为 LZ4 ``acceleration`` 传入。值越高越偏向速度，并可能降低压缩率。
     - CPU 时间敏感，或链路仅中度受限。它是默认 codec。
   * - ``zstd``
     - 通常比 LZ4 减少更多 wire bytes，但压缩和解压缩开销更高。
     - 作为 Zstandard compression level 传入。level 越高通常会用更多 CPU 时间换取更高
       压缩率。
     - 链路足够慢，减少 wire bytes 的收益高于 codec 开销。

请使用有代表性的 payload 实测后再选择。已经压缩或高熵的 tensor 可能无法缩小，而包含
大量零值或重复值的 dense tensor 可能有明显收益。LZ4 还存在单 tensor 输入大小上限；
不支持的大小会自动走 raw 路径。

配置压缩
~~~~~~~~

.. code-block:: yaml

   cluster:
     collective:
       tensor_buffer_pool:
         max_bytes: 2147483648
       tensor_compression:
         enabled: true
         codec: lz4
         level: 1
         min_bytes: 65536
         max_inflight: 2

省略 ``tensor_compression``，或设置 ``enabled: false``，即可使用原始 wire 路径。压缩
选项及默认值如下：

.. list-table:: 压缩选项
   :header-rows: 1
   :widths: 20 16 64

   * - 选项
     - 默认值
     - 含义
   * - ``enabled``
     - ``true``
     - 存在 ``tensor_compression`` 配置段时启用压缩。
   * - ``codec``
     - ``lz4``
     - 选择 ``lz4`` 或 ``zstd``。
   * - ``level``
     - ``1``
     - 设置上文所述的 codec 参数；该值必须为正数。
   * - ``min_bytes``
     - ``65536``
     - 跳过 raw byte count 小于该值的 tensor。
   * - ``max_inflight``
     - ``1``
     - 每个 Worker 分别创建该数量的 encoder instance 和 decoder instance。

``tensor_buffer_pool`` 独立于 ``tensor_compression``。它的 ``max_bytes`` 限制单个
Worker 内 active 与 cached CPU buffer 的总容量，默认值为 2 GiB。配置压缩但省略该段时，
会自动提供默认 pool。

运行与回退
~~~~~~~~~~

每个 Worker 延迟创建一个 ``TensorCodecPool`` 和一个独立的 ``TensorBufferPool``，并由其
所有 ``CollectiveGroup`` 共享。发送流程如下：

1. 尝试获取一个 encoder，且不会等待。如果所有 encoder 都在使用，本次传输保持 raw。
2. 按最坏情况输出容量从大到小排列候选 tensor，并分别尝试获取 buffer。当 codec 不支持
   该 tensor 的大小，或者预算内没有可用 buffer 时，该 tensor 保持 raw。
3. 只有压缩结果小于原始 tensor 时才使用它；否则保持 raw，并直接丢弃该 buffer，而不是
   将其放入 cache。
4. 在现有 metadata 中发送每个 tensor 的压缩大小。接收端直接将压缩 tensor 恢复到预分配
   的目标 tensor。
5. 保持压缩 payload 的 buffer lease，直到同步 payload send 完成，再将 buffer 返回 Worker
   buffer pool。

buffer pool 按 capacity 索引 idle buffer，并复用能够容纳请求的最小 size。相同 size 使用
独立 list，active 与 cached 容量共同受 ``max_bytes`` 限制。当新分配需要空间时，pool 会
从最大的 idle size bucket 开始淘汰 buffer。buffer acquisition 不会等待；因此 buffer
不可用时，该 tensor 会保持 baseline 行为。

公共依赖安装会安装这两种 codec 所需的 LZ4 和 Zstandard 系统库。请确保所有 Worker 节点
使用相同版本的 RLinf，并安装相应的 compression 依赖。


异步 API
---------------------------------

所有 P2P API 都支持异步操作，并在 ``async_op=True`` 时返回可等待的 **work handles**。  
内部实现中，提供了一个小型的层次结构：

- ``AsyncWork``：抽象基类，包含 ``wait()``、``async_wait()``、``then(func, *args, **kwargs)``、``done()``，以及链式操作辅助函数（``get_next_work()``、``get_last_work()``）。  
- ``AsyncFuncWork``：在前序任务完成时执行 Python 回调，记录一个 CUDA 事件，并可通过 ``then`` 进行链式调用。若回调返回另一个 ``AsyncWork``，则完成会延迟到链中最后的任务完成。  
- ``AsyncCollWork``：将一个 ``torch.distributed`` 的工作（如 broadcast）封装为可等待接口。它也支持 ``then`` （单一底层任务）。  
- ``AsyncChannelWork``：将 ``ray.ObjectRef`` 封装为可等待对象（用于 channel RPC）。  

关键特性：

* **等待：** ``wait()`` 为阻塞式；``async_wait()`` 适合 ``asyncio``，两者都会确保记录的 CUDA 事件完成后返回。  
* **链式调用：** ``then`` 可调度后续回调。  
* **完成检测：** ``done()`` 为非阻塞查询，用于检测底层任务是否完成。  

最小示例：

.. code-block:: python

   # 使用 await 的异步对象 send/recv
   send_work = group.send(obj, async_op=True)      # AsyncWork
   await send_work.async_wait()                    # 非阻塞等待

   recv_work = group.recv(async_op=True)           # AsyncWork
   obj = recv_work.wait()                          # 阻塞等待；返回接收到的对象

.. code-block:: python

   # 链式调用后处理步骤
   def postprocess(buf):
       # 例如：转移到 CPU、类型转换或通知其他子系统
       return None

   w = group.recv_tensor(tensor, async_op=True)    # 接收端预分配的张量
   w2 = w.then(postprocess)                        # AsyncFuncWork
   w2.wait()                                       # 确保 postprocess 完成


总结
--------------

总之，**collective** 组件为 Worker 之间的点对点数据传输提供了引擎。  
它屏蔽了 PyTorch 分布式后端的复杂细节，通过管理多个进程组来模拟 send/recv，并对 GPU 传输进行了优化。  
框架用户通常通过 `Worker.send/recv` 或 channel 操作来调用这些功能，而不是直接调用 `CollectiveGroup`。  
