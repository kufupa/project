#!/usr/bin/env bash
# Canonical MT10 task ids (metaworld.MT10().train_classes, alphabetical, metaworld==3.0.0).
# shellcheck disable=SC2034
MT10_TASK_IDS=(
  button-press-topdown-v3
  door-open-v3
  drawer-close-v3
  drawer-open-v3
  peg-insert-side-v3
  pick-place-v3
  push-v3
  reach-v3
  window-close-v3
  window-open-v3
)

mt10_task_count() {
  echo "${#MT10_TASK_IDS[@]}"
}
