# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam

device = 'cpu'
model_size = 1 * 1024**2
param = torch.nn.Parameter(torch.randn(model_size, device=device))
torch.set_printoptions(precision=3)

optimizer = DeepSpeedCPUAdam([param], lr=1e-1)
for step in range(20):
    param.grad = 1000 * torch.randn(model_size, device=device)
    optimizer.step(0)
    print(f"Optimizer Step {step}: Param[:10] = {param.data[:10]}")

optimizer.rollback(0)
print(f"Rollback Back to Step 18: Param[:10] = {param.data[:10]}")
