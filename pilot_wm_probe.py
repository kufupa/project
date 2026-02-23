import os
import numpy as np
from PIL import Image
from src.segment_grpo_loop import load_wm_bundle, _encode_state_to_latent, score_chunk_by_goal_latent

os.environ['PYTHONFAULTHANDLER'] = '1'
os.environ['JEPA_WM_DISABLE_IMAGE_HEAD'] = '1'

IMG = '/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093/frames/episode_0003/frame_000000.png'
img = np.array(Image.open(IMG).convert('RGB'))

def main() -> int:
    print('wm_probe_start', flush=True)
    wm = load_wm_bundle('/vol/bitbucket/aa6622/VGG JEPA/jepa-wms', '/vol/bitbucket/aa6622/.cache/huggingface/hub/models--facebook--jepa-wms/snapshots/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar', 'cuda')
    if wm is None:
        print('wm_load_failed')
        return 1
    print('wm_loaded', wm.planner_action_dim, wm.proprio_dim, flush=True)

    goal = _encode_state_to_latent(wm, img, np.zeros(4, dtype=np.float32))
    print('goal_shape', tuple(goal.shape), flush=True)

    chunk = np.zeros((8, wm.planner_action_dim), dtype=np.float32)
    try:
        distance, score_trace, decode_trace = score_chunk_by_goal_latent(wm, img, np.zeros(4, dtype=np.float32), chunk, goal, chunk_len=8, return_latent_trace=True)
        print('score_ok', float(distance), 'trace', len(score_trace.step_vectors), flush=True)
        return 0
    except Exception as exc:
        print(f'score_fail {type(exc).__name__}: {exc}', flush=True)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
