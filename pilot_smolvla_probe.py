import os
import numpy as np
from PIL import Image
from vendor.pi05.jepa_cem_paired_pushv3_export import _try_load_smolvla_exec, _smolvla_exec_action

os.environ['PYTHONFAULTHANDLER'] = '1'
ckpt = '/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900'
img_path = '/vol/bitbucket/aa6622/project/artifacts/phase06_oracle_baseline/run_20260411T131839Z_ep60_voracle_tpush_v3_s1000_r402093/frames/episode_0003/frame_000000.png'

def main() -> int:
    print('probe_start', flush=True)
    try:
        bundle = _try_load_smolvla_exec(ckpt, 'cuda')
    except Exception as exc:
        print(f'load_fail {type(exc).__name__}: {exc}', flush=True)
        return 2
    print('smolvla_loaded', flush=True)
    img = np.array(Image.open(img_path).convert('RGB'))

    class RenderProxy:
        def __init__(self, frame):
            self._frame = frame

        def render(self, *args, **kwargs):
            return self._frame

    obs = {'image': img, 'state': np.zeros(16, dtype=np.float32)}
    try:
        act = _smolvla_exec_action(bundle, obs, RenderProxy(img), 'push the object into the target')
        arr = np.asarray(act)
        print('action_shape', arr.shape, 'min', float(arr.min()), 'max', float(arr.max()), flush=True)
        print('done', flush=True)
        return 0
    except Exception as exc:
        print(f'act_fail {type(exc).__name__}: {exc}', flush=True)
        return 3

if __name__ == '__main__':
    raise SystemExit(main())
