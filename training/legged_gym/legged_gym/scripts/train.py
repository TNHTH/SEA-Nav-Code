# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import json
import sys
from pathlib import Path

SEA_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SEA_ROOT / "training/rsl_rl"))
sys.path.insert(0, str(SEA_ROOT / "training/legged_gym"))
from rsl_rl.runtime_preflight import preflight as shared_preflight, blocked_result

def preflight(argv):
    return shared_preflight(argv,runtime_stack="isaac_gym_preview4",repo_root=SEA_ROOT)

def main(argv=None):
    request=preflight(sys.argv[1:] if argv is None else argv)
    if request.arguments.preflight_only:
        return {"status":"preflight_passed","runtime_status":"not_executed"}
    env=None
    try:
        from rsl_rl.runtime_preflight import require_runtime_prerequisites
        require_runtime_prerequisites(request)
        import isaacgym
        import legged_gym.envs
        from legged_gym.utils import get_args,task_registry
        args=get_args(list(request.remaining_argv))
        from rsl_rl.runtime_preflight import reconcile_gym_arguments
        args=reconcile_gym_arguments(request,args)
        args.runtime_request=request
        # Validate the registered run's real constructors before the first environment.
        from rsl_rl.environment_profile import apply_algorithm_profile
        from legged_gym.utils.helpers import class_to_dict
        _,preconstruction_train_cfg=task_registry.get_cfgs(args.task)
        apply_algorithm_profile(class_to_dict(preconstruction_train_cfg),request.resolved_config)
        env,env_cfg=task_registry.make_env(name=args.task,args=args,resolved_config=request.resolved_config)
        runner,train_cfg=task_registry.make_alg_runner(env=env,name=args.task,args=args,
            log_root=str(request.paths.run_root/"training"),resolved_config=request.resolved_config)
        runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,init_at_random_ep_len=True,
                     config=env_cfg.environment_receipt)
        output=blocked_result("isaac_gym_preview4","runtime acceptance not established",
                              "complete real controller/reset and lifecycle validation")
        output["effective_environment"]=env_cfg.environment_receipt
    except (ModuleNotFoundError,ImportError) as exc:
        output=blocked_result("isaac_gym_preview4",exc,"install and lock Isaac Gym Preview 4",
                              dependency_status="unavailable")
    except Exception as exc:
        output=blocked_result("isaac_gym_preview4",exc,"resolve runtime prerequisites")
        if not str(exc).startswith("blocked:"):
            output["status"]="failed"
    finally:
        errors=[]
        if env is not None:
            for name,resource in (("viewer",getattr(env,"viewer",None)),("sim",getattr(env,"sim",None))):
                if resource is not None:
                    try:
                        getattr(env.gym,"destroy_"+name)(resource)
                    except Exception as exc:
                        errors.append(dict(resource=name,error=str(exc)))
    if errors:
        output.update(status="failed",close_errors=errors)
    output["ok"]=False
    from rsl_rl.runtime_preflight import publish_runtime_result
    return publish_runtime_result(output,request)

if __name__ == "__main__":
    output=main()
    print(json.dumps(output,sort_keys=True))
    raise SystemExit(0 if output["status"]=="preflight_passed" else 3)
